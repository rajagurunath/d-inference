package batchlane

// dispatcher.go is the 1 Hz control loop (docs/design/tidal-batch-lane.md §3.4).
// One tick, in order:
//
//	drain   settle the outcomes the previous tick's dispatch goroutines reported
//	observe smooth every slot's live signal with an EWMA
//	control run the per-slot AIMD, producing an in-flight budget per model and
//	        fleet-wide
//	rank    laxity → urgency → priority for every open batch
//	claim   Σ(target − inflight) items, highest priority first, line order within
//	        a batch, only from in_progress batches, each capped at its own
//	        model's headroom
//	dispatch one goroutine per claimed item under its batch's context
//	sweep   expire batches past their window, drain cancelled ones
//
// Tick is synchronous apart from the dispatch goroutines, which report back
// through a channel drained at the start of the next tick, so a test can drive
// the whole loop by hand with no sleeping and no wall clock.
//
// Placement is NOT decided here. The dispatcher decides only HOW MANY batch
// rows may be in flight, per model and fleet-wide; WHERE each one lands is the
// reservation path's decision, which admits a LaneBatch request only to a slot
// with no waiting row and running below its batch allowance
// (registry/scheduler.go).
//
// Privacy: nothing in this file logs a request body, a result, a custom_id or a
// metadata value. Log fields are ids, counts, targets and bounded error codes.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/eigeninference/d-inference/coordinator/saferun"
	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

// Outcome is the result of running one batch item through the dispatch funnel.
// It mirrors api.BatchOutcome field for field; batchlane must not import api
// (api imports batchlane), so the api layer adapts one to the other when it
// builds the DispatchFn.
type Outcome struct {
	RequestID                      string
	PromptTokens, CompletionTokens int
	ResponseBody                   []byte
	// ErrCode is "" on success, ErrCodeNoCapacity when no slot had batch
	// headroom, ErrCodeCancelled when the attempt's context was cancelled, and
	// ErrCodeRequestFailed for every other non-success terminal.
	ErrCode string
}

// The bounded outcome vocabulary. Neither the store nor a log line ever carries
// anything else, so provider prose can never reach a consumer's error file.
const (
	// ErrCodeNoCapacity means no provider slot had batch headroom. The claim is
	// released without charging an attempt: the item never reached a provider.
	ErrCodeNoCapacity = "no_capacity"
	// ErrCodeCancelled means the attempt's context was cancelled — coordinator
	// shutdown, or the batch itself being cancelled or expired. Accounted like
	// ErrCodeNoCapacity: no attempt is charged, because the failure is ours.
	ErrCodeCancelled = "cancelled"
	// ErrCodeRequestFailed is every other non-success terminal.
	ErrCodeRequestFailed = "request_failed"
)

// DispatchFn runs one batch item through the standard consumer dispatch funnel
// on the batch lane and waits for it to complete. api wires
// (*api.Server).DispatchBatchItem to it.
type DispatchFn func(ctx context.Context, accountID, apiKeyID, model string, body []byte) (Outcome, error)

// FinalizeFn assembles a batch's output and error files once it has no open
// items left, and moves it to its terminal status. PR2's assembler provides the
// real hook; nil is a no-op, which leaves a completed batch's items settled and
// its files unwritten.
type FinalizeFn func(batchID string, now time.Time) error

// Config are the dispatcher's tunables. The zero value means the spec defaults.
type Config struct {
	// Tick is the control period. <= 0 means DefaultTick.
	Tick time.Duration
	// MaxAttempts is how many times one item may reach a provider before it is
	// settled as failed. <= 0 means DefaultMaxAttempts.
	MaxAttempts int
	// AIMD are the per-slot controller's watermarks.
	AIMD AIMDConfig
	// OutputRetention is how long an item's sealed result blob survives after
	// its batch finalizes. <= 0 means DefaultOutputRetention. The wiring passes
	// api.BatchOutputRetention so the item blobs and the assembled files expire
	// on one boundary.
	OutputRetention time.Duration
	// PurgeInterval is how often the sweep runs Purge. <= 0 means
	// DefaultPurgeInterval.
	PurgeInterval time.Duration
	// OrphanInterval is how often the sweep walks the blob directory for item
	// blobs no row references. <= 0 means DefaultOrphanInterval.
	OrphanInterval time.Duration
	// Purge is the file-retention pass — api's PurgeExpiredBatchFiles, which
	// deletes the assembled output/error blobs and marks their rows purged. It
	// is a Config field rather than a New parameter to keep New's signature the
	// one the plan specifies. nil skips the pass.
	Purge func(now time.Time) (int, error)
}

const (
	// DefaultTick is the 1 Hz control period from the spec.
	DefaultTick = time.Second
	// DefaultMaxAttempts is the spec's maxAttempts.
	DefaultMaxAttempts = 3
	// DefaultOutputRetention is how long an item's sealed result blob survives
	// after its batch finalizes. It mirrors api.BatchOutputRetention, which the
	// wiring passes in; batchlane cannot import api to read the constant.
	DefaultOutputRetention = 7 * 24 * time.Hour
	// DefaultPurgeInterval is how often the sweep runs the file retention pass.
	// Retention is a 7-day boundary, so a minute of slack costs nothing and
	// keeps the query off the hot path of a busy tick.
	DefaultPurgeInterval = time.Minute
	// DefaultOrphanInterval is how often the sweep walks the blob directory
	// looking for item blobs no row references. A full listing is the most
	// expensive thing the dispatcher does, and the condition it repairs is a
	// crash between sealing an item body and committing its rows, so hourly is
	// far more often than it can occur.
	DefaultOrphanInterval = time.Hour
	// maxOrphanDeletes bounds the unlinks one orphan pass performs, so a
	// directory that somehow filled with stale blobs is drained over several
	// passes instead of blocking on tens of thousands of unlinks.
	maxOrphanDeletes = 1000
	// maxOrphanScan bounds the store probes one orphan pass performs. The
	// delete bound alone does not bound the pass: a directory full of blobs
	// that all still have rows produces one BatchItemExists per blob and no
	// deletes at all, so a large store would issue a query per blob per hour
	// forever. A pass that hits either bound stops and the next one continues.
	maxOrphanScan = 2000
	// itemBlobPrefix is the id prefix every batch-item blob ref carries. File
	// blobs use "file-" and are owned by the file retention pass, so the orphan
	// sweep never looks at one.
	itemBlobPrefix = "bitem_"
	// itemInputBlobSuffix is what api.BatchItemInputRef appends to an item id.
	// The result ref is the bare item id, so stripping this suffix (when
	// present) turns either ref back into the id to probe for.
	itemInputBlobSuffix = "-in"
	// resultBuffer bounds the outcomes a tick may have to drain. Dispatch
	// goroutines are themselves bounded by the AIMD targets, which are bounded
	// by the fleet's batch row allowance, so this is slack rather than a limit.
	resultBuffer = 1024
)

// ResultBlobRef is the sealed blob key one item's result is written under. It
// MUST equal api.BatchItemResultRef, which the assembler falls back to when an
// item row carries no ref; api/batch_ref_agreement_test.go pins the two
// together, because batchlane cannot import api to share the function.
//
// The item's sealed REQUEST body lives under a different ref (api's
// BatchItemInputRef), which the dispatcher never derives: it reads the ref off
// the item row, so the input key rule stays entirely PR2's.
func ResultBlobRef(itemID string) string { return itemID }

// Dispatcher fills idle provider slots with batch work. One instance per
// coordinator, run under saferun.
type Dispatcher struct {
	st       store.Store
	blob     *sealedblob.Store
	view     RegistryView
	dispatch DispatchFn
	finalize FinalizeFn
	cfg      Config
	logger   *slog.Logger

	results chan itemOutcome
	wg      sync.WaitGroup
	// stop is closed when Run returns, so a dispatch goroutine reporting into a
	// full channel after shutdown exits instead of leaking.
	stop     chan struct{}
	stopOnce sync.Once

	mu       sync.Mutex
	slots    map[SlotKey]*slotState
	batches  map[string]*batchState
	attempts map[string]int       // per item id, in memory (see settleFailure)
	retire   map[string]time.Time // batch id -> when its result blobs may go
	inflight int
	// inflightByModel is the fleet-wide count split by the model a batch
	// declares, so a model's budget is spent only on that model's items. Only
	// batches that carry a model are counted; a file-form batch declares none
	// (its lines each carry their own) and spends the fleet budget.
	inflightByModel map[string]int
	lastPurge       time.Time
	lastOrphan      time.Time
	// orphanRunning guards the off-tick orphan pass, so a pass that outlives its
	// own interval is never joined by a second one.
	orphanRunning bool
}

// slotState is one provider·slot's controller plus the smoothing that feeds it.
type slotState struct {
	aimd   *AIMD
	decode EWMA
	kv     EWMA
}

// batchState is the per-batch escalation state and the cancellation handle for
// everything the dispatcher has in flight for it.
type batchState struct {
	rate     ObservedRate
	bucket   TokenBucket
	ctx      context.Context
	cancel   context.CancelFunc
	inflight int
	// claimable caches the last tick's view of whether the batch may be claimed
	// from, so a settle can tell "release this back to pending" from "leave it
	// to the sweep" without a second store read.
	claimable bool
}

// itemOutcome is what a dispatch goroutine reports back.
type itemOutcome struct {
	// batch is the row the tick claimed this item from, carried through
	// claim → dispatch → settle so settle never has to resolve it again. That
	// re-read is what used to downgrade a consumer-sealed result: a batch that
	// had just left the open list resolved to nothing, and the result was
	// written under the coordinator's own key instead of the consumer's.
	batch   *store.Batch
	batchID string
	itemID  string
	// model is the batch's declared model, "" for the file form. It is the key
	// the per-model in-flight count was charged under at claim time, carried
	// back so settle credits the same one.
	model   string
	outcome Outcome
	err     error
}

// errDispatchPanicked is the error a recovered dispatch goroutine settles with.
// Paired with ErrCodeRequestFailed it reads as permanent, so the claim is
// failed rather than stranded inflight until the batch expires.
var errDispatchPanicked = errors.New("batch item dispatch panicked")

// New builds a dispatcher and performs restart recovery: every item some
// previous process left inflight is returned to pending, keeping its attempt
// count, because the dispatch really did happen and only its outcome is
// unknown.
//
// finalize may be nil (no-op). A nil blob store or dispatch funnel makes Tick a
// no-op rather than a panic, so a coordinator with the batch lane unconfigured
// can still start the loop.
func New(
	st store.Store,
	blob *sealedblob.Store,
	view RegistryView,
	dispatch DispatchFn,
	finalize FinalizeFn,
	cfg Config,
	logger *slog.Logger,
) *Dispatcher {
	if logger == nil {
		logger = slog.Default()
	}
	if cfg.Tick <= 0 {
		cfg.Tick = DefaultTick
	}
	if cfg.MaxAttempts <= 0 {
		cfg.MaxAttempts = DefaultMaxAttempts
	}
	if cfg.OutputRetention <= 0 {
		cfg.OutputRetention = DefaultOutputRetention
	}
	if cfg.PurgeInterval <= 0 {
		cfg.PurgeInterval = DefaultPurgeInterval
	}
	if cfg.OrphanInterval <= 0 {
		cfg.OrphanInterval = DefaultOrphanInterval
	}
	d := &Dispatcher{
		st:       st,
		blob:     blob,
		view:     view,
		dispatch: dispatch,
		finalize: finalize,
		cfg:      cfg,
		logger:   logger,
		results:  make(chan itemOutcome, resultBuffer),
		stop:     make(chan struct{}),
		slots:    map[SlotKey]*slotState{},
		batches:  map[string]*batchState{},
		attempts: map[string]int{},
		retire:   map[string]time.Time{},

		inflightByModel: map[string]int{},
	}
	d.requeueAfterRestart()
	return d
}

// requeueAfterRestart returns every open batch's inflight items to pending.
func (d *Dispatcher) requeueAfterRestart() {
	if d.st == nil {
		return
	}
	batches, err := d.st.ListOpenBatches()
	if err != nil {
		d.logger.Error("batch lane: could not list open batches for restart recovery", "error", err)
		return
	}
	total := 0
	for _, b := range batches {
		n, err := d.st.RequeueInflightItems(b.ID)
		if err != nil {
			d.logger.Error("batch lane: could not requeue inflight items", "batch_id", b.ID, "error", err)
			continue
		}
		total += n
	}
	if total > 0 {
		d.logger.Info("batch lane: requeued items left inflight by a previous process",
			"batches", len(batches), "items", total)
	}
}

// Run drives Tick at cfg.Tick until ctx is done, then cancels every in-flight
// item and waits for the dispatch goroutines to report. Call it under
// saferun.Go.
func (d *Dispatcher) Run(ctx context.Context) {
	ticker := time.NewTicker(d.cfg.Tick)
	defer ticker.Stop()
	d.logger.Info("batch lane dispatcher started", "tick", d.cfg.Tick, "max_attempts", d.cfg.MaxAttempts)

	for {
		select {
		case <-ctx.Done():
			d.shutdown()
			return
		case now := <-ticker.C:
			d.Tick(ctx, now)
		}
	}
}

// shutdown cancels every batch context and waits for the goroutines. Items left
// inflight in the store are recovered by the next process's New.
func (d *Dispatcher) shutdown() {
	d.mu.Lock()
	for _, bs := range d.batches {
		bs.cancel()
	}
	d.mu.Unlock()
	d.stopOnce.Do(func() { close(d.stop) })
	d.wg.Wait()
	d.logger.Info("batch lane dispatcher stopped")
}

// Tick runs one iteration of the control loop. Everything it decides is a
// function of (state, the view's signals, now); it never reads the clock.
func (d *Dispatcher) Tick(ctx context.Context, now time.Time) {
	if d.st == nil || d.blob == nil || d.dispatch == nil || d.view == nil {
		return
	}

	d.drainResults(now)

	budget := d.updateTargets(now)

	batches, err := d.st.ListOpenBatches()
	if err != nil {
		d.logger.Error("batch lane: could not list open batches", "error", err)
		return
	}

	d.claimAndDispatch(ctx, batches, budget, now)
	d.sweep(batches, now)
	d.pruneBatchState(batches, now)
	d.retention(now)
}

// laneBudget is one tick's in-flight allowance. The AIMD runs per slot, so the
// allowance a batch may actually use is the one its OWN model's slots produced:
// a batch for model X that spent the fleet's budget would claim items no X slot
// can take, and every one of them would run the full dispatch funnel — a fleet
// scan under the registry's read lock — only to come back no_capacity and be
// released. fleet is the sum over every slot, for the file form, whose lines
// each carry their own model and which therefore has no single model to scope
// to.
type laneBudget struct {
	fleet    int
	perModel map[string]int
}

// forModel is the ceiling a batch declaring model may claim under. A batch with
// no declared model (the file form) is bounded by the fleet budget alone.
func (b laneBudget) forModel(model string) int {
	if model == "" {
		return b.fleet
	}
	if n := b.perModel[model]; n < b.fleet {
		return n
	}
	return b.fleet
}

// spend debits a claim of n items from the fleet budget and, when the batch
// declared a model, from that model's headroom. Neither goes below zero.
func (b *laneBudget) spend(model string, n int) {
	if b.fleet -= n; b.fleet < 0 {
		b.fleet = 0
	}
	if model == "" {
		return
	}
	if left := b.perModel[model] - n; left > 0 {
		b.perModel[model] = left
	} else {
		b.perModel[model] = 0
	}
}

// updateTargets folds each slot's fresh signal into its EWMAs, runs its AIMD
// step, and returns the in-flight budget: Σtarget − items already out,
// fleet-wide and per model. A slot the view no longer reports has disconnected
// and its controller state is dropped.
func (d *Dispatcher) updateTargets(now time.Time) laneBudget {
	signals := d.view.Slots("")

	d.mu.Lock()
	defer d.mu.Unlock()

	total := 0
	byModel := make(map[string]int, len(signals))
	for key, sig := range signals {
		st, ok := d.slots[key]
		if !ok {
			st = &slotState{aimd: NewAIMD(d.cfg.AIMD)}
			d.slots[key] = st
		}
		// An unmeasured decode rate (0) is not a slow one: folding it in would
		// drag the average to zero and pin the target at the floor forever.
		if sig.DecodeTPS > 0 {
			sig.DecodeTPS = st.decode.Observe(sig.DecodeTPS)
		} else if st.decode.Initialized() {
			sig.DecodeTPS = st.decode.Value
		}
		// A slot with no published token budget has no KV sample to fold in;
		// folding a placeholder would poison the average for the ticks after the
		// provider does start reporting one.
		if sig.KVKnown {
			sig.KV = st.kv.Observe(sig.KV)
		}
		// Waiting is deliberately NOT smoothed: it is the backpressure signal,
		// and an EWMA of it would both delay the backoff and never fully decay,
		// leaving the target halved long after the online burst passed.
		target := st.aimd.Update(sig)
		total += target
		byModel[key.Model] += target
	}
	for key := range d.slots {
		if _, ok := signals[key]; !ok {
			delete(d.slots, key)
		}
	}

	budget := laneBudget{fleet: total - d.inflight, perModel: byModel}
	if budget.fleet < 0 {
		budget.fleet = 0
	}
	// A model whose in-flight items already meet its slots' targets has no
	// headroom, and a model with no slots at all has none either — the map is
	// keyed on the slots the view reported, so an absent model reads as 0.
	for model := range byModel {
		if byModel[model] -= d.inflightByModel[model]; byModel[model] < 0 {
			byModel[model] = 0
		}
	}
	return budget
}

// batchPlan is one open batch's rank for this tick.
type batchPlan struct {
	batch   *store.Batch
	pending int
	urgency float64
	prio    int
}

// claimAndDispatch ranks the claimable batches by priority, spends the budget
// across them highest priority first, and starts one goroutine per claimed
// item. Each batch is capped at its own model's headroom as well as at what is
// left of the fleet budget. A batch whose slack has run out gets a token-bucket
// progress floor of one item even when the budget is zero.
func (d *Dispatcher) claimAndDispatch(ctx context.Context, batches []*store.Batch, budget laneBudget, now time.Time) {
	plans := make([]batchPlan, 0, len(batches))
	for _, b := range batches {
		// Only in_progress batches are claimable, and never one already past
		// its window: this tick's sweep is about to expire it.
		claimable := b.Status == store.BatchInProgress && now.Before(b.ExpiresAt)
		bs := d.batchStateFor(ctx, b.ID)
		d.mu.Lock()
		bs.claimable = claimable
		d.mu.Unlock()
		if !claimable {
			continue
		}

		_, pending, inflight, _, _, err := d.st.CountItems(b.ID)
		if err != nil {
			d.logger.Error("batch lane: could not count items", "batch_id", b.ID, "error", err)
			continue
		}
		if pending == 0 {
			continue
		}
		u := d.urgencyOf(bs, b, pending+inflight, now)
		plans = append(plans, batchPlan{batch: b, pending: pending, urgency: u, prio: Priority(u)})
	}

	// Highest priority (lowest number) first; oldest batch breaks a tie so a
	// fleet of equally healthy batches drains in submission order.
	sort.SliceStable(plans, func(i, j int) bool {
		if plans[i].prio != plans[j].prio {
			return plans[i].prio < plans[j].prio
		}
		if !plans[i].batch.CreatedAt.Equal(plans[j].batch.CreatedAt) {
			return plans[i].batch.CreatedAt.Before(plans[j].batch.CreatedAt)
		}
		return plans[i].batch.ID < plans[j].batch.ID
	})

	// The floor is granted at most once per tick FLEET-WIDE, not once per urgent
	// batch: a hundred batches crossing FloorUrgency together must not put a
	// hundred rows on a fleet whose AIMD target is zero. The plans are already
	// ordered by priority, so the grant goes to the batch closest to missing its
	// window; a batch whose own bucket is empty passes the grant on rather than
	// consuming it.
	floorGranted := false
	for _, p := range plans {
		model := p.batch.Model
		want := p.pending
		if headroom := budget.forModel(model); want > headroom {
			want = headroom
		}
		if want == 0 && !floorGranted && p.urgency >= FloorUrgency {
			// The deadline progress floor: one item, rate limited per batch to
			// FloorItemsPerSec, so an urgent batch is never starved to expiry
			// however busy the online lane is. It does NOT raise the AIMD
			// target — nor the model's headroom — because the reservation path
			// still refuses a slot with no headroom, so the floor can only use
			// capacity that really exists. One item per tick fleet-wide is the
			// bound on what a floor grant for a model with no slots can cost.
			bs := d.batchStateFor(ctx, p.batch.ID)
			d.mu.Lock()
			granted := bs.bucket.TryTake(now)
			d.mu.Unlock()
			if granted {
				want = 1
				floorGranted = true
			}
		}
		if want == 0 {
			continue
		}
		claimed, err := d.st.ClaimPendingItems(p.batch.ID, want, now)
		if err != nil {
			d.logger.Error("batch lane: could not claim items", "batch_id", p.batch.ID, "error", err)
			continue
		}
		if len(claimed) == 0 {
			continue
		}
		budget.spend(model, len(claimed))
		d.logger.Debug("batch lane: claimed items",
			"batch_id", p.batch.ID, "items", len(claimed), "priority", p.prio,
			"budget_left", budget.fleet, "model_budget_left", budget.forModel(model))
		for _, it := range claimed {
			d.start(ctx, p.batch, it)
		}
	}
}

// urgencyOf computes a batch's urgency from its own observed completion rate,
// falling back to the fleet-wide rate and then to pure slack.
func (d *Dispatcher) urgencyOf(bs *batchState, b *store.Batch, remaining int, now time.Time) float64 {
	d.mu.Lock()
	rate, known := bs.rate.PerSec(now)
	d.mu.Unlock()
	if !known {
		rate, known = d.st.CompletionRate(ObservedRateWindow, now)
	}
	if !known {
		rate = 0 // cold start: slack only
	}
	return Urgency(Laxity(b.ExpiresAt, now, remaining, rate))
}

// start launches one item's dispatch under its batch's context.
func (d *Dispatcher) start(ctx context.Context, b *store.Batch, it *store.BatchItem) {
	bs := d.batchStateFor(ctx, b.ID)

	d.mu.Lock()
	d.inflight++
	if b.Model != "" {
		d.inflightByModel[b.Model]++
	}
	bs.inflight++
	itemCtx := bs.ctx
	d.mu.Unlock()

	itemID, blobRef := it.ID, it.BlobRef
	// The key id travels on the batch row (PR3c), so batch work is attributed to
	// the key that submitted it and its AllowedModels and spend cap apply.
	accountID, apiKeyID, batchModel := b.AccountID, b.APIKeyID, b.Model

	// res is primed with the permanent failure a panicking dispatch would leave
	// behind, so a recovered goroutine still settles its claim. The batch row is
	// carried along: settle must never re-resolve it (a consumer-sealed result
	// would silently fall back to the coordinator's key).
	res := itemOutcome{
		batch:   b,
		batchID: b.ID,
		itemID:  itemID,
		model:   batchModel,
		outcome: Outcome{ErrCode: ErrCodeRequestFailed},
		err:     errDispatchPanicked,
	}

	d.wg.Add(1)
	go func() {
		defer d.wg.Done()
		// Defers run LIFO, so the report below runs AFTER saferun.Recover has
		// swallowed a panic — which is what turns a panicking dispatch into a
		// settled item instead of an item stranded inflight until expiry.
		defer func() {
			select {
			case d.results <- res:
			case <-d.stop:
				// Shutting down. The item stays inflight in the store and the
				// next process's restart recovery returns it to pending.
			}
		}()
		defer saferun.Recover(d.logger, "batch_dispatch_item")

		res.outcome, res.err = d.runItem(itemCtx, accountID, apiKeyID, batchModel, blobRef)
	}()
}

// runItem opens the sealed request body and runs it through the funnel. The
// plaintext lives only for the duration of this call: it is never stored on the
// Dispatcher, never logged, and never copied into an outcome.
func (d *Dispatcher) runItem(ctx context.Context, accountID, apiKeyID, batchModel, blobRef string) (Outcome, error) {
	// A body that cannot be opened or parsed is a permanent failure for the
	// item: the pairing of a non-nil error with ErrCodeRequestFailed is what
	// tells settle not to spend the retry budget on it.
	body, err := d.blob.Open(blobRef)
	if err != nil {
		return Outcome{ErrCode: ErrCodeRequestFailed}, fmt.Errorf("open item body: %w", err)
	}
	model := batchModel
	if model == "" {
		model, err = modelOf(body)
		if err != nil {
			return Outcome{ErrCode: ErrCodeRequestFailed}, err
		}
	}
	// The batch's submitting API key rides with every one of its items, so the
	// key's AllowedModels and spend cap are enforced on batch work. "" means the
	// batch was created without one (Privy JWT, admin key) and runs under
	// account-level limits only.
	return d.dispatch(ctx, accountID, apiKeyID, model, body)
}

// modelOf reads the model field out of an item's request body. The inline batch
// form carries the model on the batch instead; the file form carries it per
// line, which is what OpenAI's wire format specifies.
func modelOf(body []byte) (string, error) {
	var envelope struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		// The error deliberately does not wrap the decoder's message, which
		// would quote the offending bytes.
		return "", errors.New("batch item body is not a JSON object")
	}
	if envelope.Model == "" {
		return "", errors.New("batch item body carries no model")
	}
	return envelope.Model, nil
}

// drainResults settles every outcome reported since the last tick.
func (d *Dispatcher) drainResults(now time.Time) {
	for {
		select {
		case res := <-d.results:
			d.settle(res, now)
		default:
			return
		}
	}
}

// settle applies one outcome. A result for an item that is no longer inflight
// (expired, cancelled, or already settled) is ignored: FinishItem and
// ReleaseItem both return false and change nothing.
func (d *Dispatcher) settle(res itemOutcome, now time.Time) {
	d.mu.Lock()
	d.inflight--
	if d.inflight < 0 {
		d.inflight = 0
	}
	if res.model != "" {
		if left := d.inflightByModel[res.model] - 1; left > 0 {
			d.inflightByModel[res.model] = left
		} else {
			delete(d.inflightByModel, res.model)
		}
	}
	bs := d.batches[res.batchID]
	if bs != nil && bs.inflight > 0 {
		bs.inflight--
	}
	claimable := bs != nil && bs.claimable
	d.mu.Unlock()

	switch {
	case res.err == nil && res.outcome.ErrCode == "":
		d.settleSuccess(res, claimable, now)
	case res.outcome.ErrCode == ErrCodeNoCapacity || res.outcome.ErrCode == ErrCodeCancelled:
		// Neither is the item's fault, so no attempt is charged. If the batch
		// is no longer claimable its items belong to the sweep, which has
		// already moved them to expired or cancelled.
		if !claimable {
			// The item is terminal now, so its retry tally goes with it —
			// leaving the entry behind leaks one map slot per item for the life
			// of the process.
			d.forgetAttempts(res.itemID)
			return
		}
		if _, err := d.st.ReleaseItem(res.itemID); err != nil {
			d.logger.Error("batch lane: could not release a claim",
				"batch_id", res.batchID, "item_id", res.itemID, "code", res.outcome.ErrCode, "error", err)
		}
	default:
		// A non-nil error carrying ErrCodeRequestFailed is the funnel saying the
		// failure is PERMANENT for this item — an unusable API key, a body it
		// cannot parse, a blob it cannot open. Retrying would burn the batch's
		// whole attempt budget on an outcome that cannot change.
		permanent := res.err != nil && res.outcome.ErrCode == ErrCodeRequestFailed
		d.settleFailure(res, claimable, permanent, now)
	}
}

// settleSuccess seals the response and moves the item to succeeded.
//
// Two guards come before the blob is written, because a result blob written
// under the wrong key — or written at all for an item that is already
// terminal — is worse than no result:
//
//   - a settle with no batch row cannot know which key the consumer asked for,
//     so it fails the item permanently rather than sealing to the coordinator's
//     own key. Downgrading would hand the coordinator plaintext the consumer
//     paid to keep from it;
//   - a settle for a batch the sweep has already closed writes nothing at all.
//     FinishItem would refuse the item anyway and the blob would be deleted on
//     the next line; skipping the write is the same outcome without the round
//     trip, and it leaves an expired batch with nothing on disk.
func (d *Dispatcher) settleSuccess(res itemOutcome, claimable bool, now time.Time) {
	if res.batch == nil {
		d.logger.Error("batch lane: a settle arrived with no batch row",
			"batch_id", res.batchID, "item_id", res.itemID)
		d.settleFailure(res, claimable, true, now)
		return
	}
	if !claimable {
		d.forgetAttempts(res.itemID)
		return
	}

	ref := ResultBlobRef(res.itemID)
	if err := d.putResult(res.batch, ref, res.outcome.ResponseBody); err != nil {
		d.logger.Error("batch lane: could not seal a result",
			"batch_id", res.batchID, "item_id", res.itemID, "error", err)
		// The response cannot be stored, so it can never be assembled: retrying
		// the dispatch would only reproduce the same failure at a provider's
		// expense.
		d.settleFailure(res, true, true, now)
		return
	}

	ok, err := d.st.FinishItem(store.ItemResult{
		ItemID:           res.itemID,
		Succeeded:        true,
		PromptTokens:     res.outcome.PromptTokens,
		CompletionTokens: res.outcome.CompletionTokens,
		RequestID:        res.outcome.RequestID,
		ResultBlobRef:    ref,
	}, now)
	if err != nil {
		// A FinishItem error does NOT prove the finish did not land: an error
		// raised at commit (or on the way back from one) can follow a
		// transaction that already committed, leaving a succeeded row pointing
		// at this ref. ReleaseItem is the discriminator — it moves an INFLIGHT
		// row back to pending and returns true only then. Only a true release
		// proves nothing references the blob, and only then may it be dropped
		// so the next tick re-dispatches cleanly. A false release means the
		// finish may have committed, so the blob stays: an orphan blob costs
		// disk until the sweep, a deleted one costs the consumer the result.
		d.logger.Error("batch lane: could not finish an item",
			"batch_id", res.batchID, "item_id", res.itemID, "error", err)
		released, rerr := d.st.ReleaseItem(res.itemID)
		if rerr != nil {
			d.logger.Error("batch lane: could not re-offer an unfinishable item",
				"batch_id", res.batchID, "item_id", res.itemID, "error", rerr)
		}
		if !released {
			// The item was not inflight any more, so the finish may have
			// committed. Keep the blob, drop the retry tally with the item, and
			// give finalize its chance in case that commit closed the batch.
			d.logger.Warn("batch lane: keeping a result blob for an item that could not be re-offered",
				"batch_id", res.batchID, "item_id", res.itemID)
			d.forgetAttempts(res.itemID)
			d.runFinalize(res.batchID, now)
			return
		}
		if derr := d.blob.Delete(ref); derr != nil && !errors.Is(derr, sealedblob.ErrNotFound) {
			d.logger.Error("batch lane: could not drop an unfinished result blob",
				"batch_id", res.batchID, "item_id", res.itemID, "error", derr)
		}
		return
	}
	if !ok {
		// A late result for an item the sweep already closed. Drop the blob we
		// just wrote so an expired batch leaves nothing behind.
		if err := d.blob.Delete(ref); err != nil && !errors.Is(err, sealedblob.ErrNotFound) {
			d.logger.Error("batch lane: could not drop a late result blob",
				"batch_id", res.batchID, "item_id", res.itemID, "error", err)
		}
		d.forgetAttempts(res.itemID)
		return
	}

	d.mu.Lock()
	delete(d.attempts, res.itemID)
	if bs := d.batches[res.batchID]; bs != nil {
		bs.rate.Record(now)
	}
	d.mu.Unlock()

	d.runFinalize(res.batchID, now)
}

// forgetAttempts drops one item's in-memory retry tally.
func (d *Dispatcher) forgetAttempts(itemID string) {
	d.mu.Lock()
	delete(d.attempts, itemID)
	d.mu.Unlock()
}

// putResult seals the response body to the consumer's key when the batch
// carries one, and to the coordinator's own key otherwise. b is the row the
// tick claimed the item from — never a re-read, so a batch that left the open
// list mid-dispatch cannot silently downgrade a consumer-sealed result.
func (d *Dispatcher) putResult(b *store.Batch, ref string, body []byte) error {
	if b.ResultPublicKey != "" {
		key, err := decodePublicKey(b.ResultPublicKey)
		if err != nil {
			return err
		}
		return d.blob.PutTo(ref, body, key)
	}
	return d.blob.PutPlain(ref, body)
}

// decodePublicKey parses a batch's base64 X25519 result key. The error never
// quotes the value.
func decodePublicKey(encoded string) ([32]byte, error) {
	var key [32]byte
	raw, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return key, errors.New("result_public_key is not valid base64")
	}
	if len(raw) != 32 {
		return key, fmt.Errorf("result_public_key must be 32 bytes, got %d", len(raw))
	}
	copy(key[:], raw)
	return key, nil
}

// settleFailure charges an attempt and either re-offers the item or settles it
// as failed.
//
// The attempt tally is kept in memory rather than in the item row because the
// store's only per-item requeue, ReleaseItem, un-counts the claim's attempt by
// design (it exists for the no-capacity path), and the batch-wide
// RequeueInflightItems would yank items that are genuinely still running. A
// coordinator restart therefore forgets a partial retry budget, which costs at
// most MaxAttempts-1 extra attempts per item and never loses one.
func (d *Dispatcher) settleFailure(res itemOutcome, claimable, permanent bool, now time.Time) {
	d.mu.Lock()
	d.attempts[res.itemID]++
	attempts := d.attempts[res.itemID]
	d.mu.Unlock()

	if !permanent && attempts < d.cfg.MaxAttempts && claimable {
		if _, err := d.st.ReleaseItem(res.itemID); err != nil {
			d.logger.Error("batch lane: could not re-offer a failed item",
				"batch_id", res.batchID, "item_id", res.itemID, "attempts", attempts, "error", err)
		}
		return
	}

	d.mu.Lock()
	delete(d.attempts, res.itemID)
	d.mu.Unlock()

	ok, err := d.st.FinishItem(store.ItemResult{
		ItemID:    res.itemID,
		Succeeded: false,
		ErrorCode: ErrCodeRequestFailed,
	}, now)
	if err != nil {
		d.logger.Error("batch lane: could not fail an item",
			"batch_id", res.batchID, "item_id", res.itemID, "error", err)
		return
	}
	if !ok {
		return // already terminal
	}
	d.logger.Info("batch lane: item failed",
		"batch_id", res.batchID, "item_id", res.itemID,
		"attempts", attempts, "permanent", permanent, "code", ErrCodeRequestFailed)
	d.runFinalize(res.batchID, now)
}

// sweep expires batches past their completion window and drains cancelled ones.
func (d *Dispatcher) sweep(batches []*store.Batch, now time.Time) {
	for _, b := range batches {
		switch {
		case b.Status == store.BatchInProgress && !now.Before(b.ExpiresAt):
			d.expire(b, now)
		case b.Status == store.BatchCancelling:
			d.drainCancelled(b, now)
		}
	}
}

// expire closes a batch that ran out of window. Expired items move neither
// counts_completed nor counts_failed, so completed + failed stays ≤ total.
//
// The terminal transition belongs to finalize, not here: finalize assembles the
// output and error files (an expired item becomes an error line carrying
// batch_expired) and only then CASes in_progress → expired. Closing the batch
// first would hand finalize a batch that is no longer open, and the consumer
// would get an expired batch with no files at all.
func (d *Dispatcher) expire(b *store.Batch, now time.Time) {
	d.cancelBatch(b.ID)
	n, err := d.st.ExpireOpenItems(b.ID, now)
	if err != nil {
		d.logger.Error("batch lane: could not expire open items", "batch_id", b.ID, "error", err)
		return
	}
	d.logger.Info("batch lane: batch window closed", "batch_id", b.ID, "expired_items", n)
	d.closeBatch(b, store.BatchInProgress, store.BatchExpired, now)
}

// drainCancelled stops a cancelling batch's in-flight work and closes it once
// nothing is outstanding. Items are cancelled immediately so a result that
// lands after this point is ignored by FinishItem; as in expire, finalize
// performs the cancelling → cancelled transition once the files are attached.
func (d *Dispatcher) drainCancelled(b *store.Batch, now time.Time) {
	d.cancelBatch(b.ID)
	if _, err := d.st.CancelOpenItems(b.ID, now); err != nil {
		d.logger.Error("batch lane: could not cancel open items", "batch_id", b.ID, "error", err)
		return
	}
	d.closeBatch(b, store.BatchCancelling, store.BatchCancelled, now)
}

// closeBatch hands the terminal transition to finalize, which performs it after
// attaching the assembled files. A crash before that leaves the batch open with
// its items already terminal, and the next tick retries the whole pass.
//
// With no finalize hook wired there is no assembler and so nothing to order
// against, and the batch would otherwise sit open forever: the dispatcher
// performs the CAS itself in that case.
func (d *Dispatcher) closeBatch(b *store.Batch, from, to store.BatchStatus, now time.Time) {
	if d.finalize != nil {
		d.runFinalize(b.ID, now)
		return
	}
	ok, err := d.st.SetBatchStatus(b.ID, from, to, now)
	if err != nil {
		d.logger.Error("batch lane: could not close a batch",
			"batch_id", b.ID, "status", string(to), "error", err)
		return
	}
	if ok {
		d.logger.Info("batch lane: batch closed without an assembler",
			"batch_id", b.ID, "status", string(to))
	}
}

// cancelBatch cancels every dispatch the batch has in flight.
func (d *Dispatcher) cancelBatch(batchID string) {
	d.mu.Lock()
	bs := d.batches[batchID]
	d.mu.Unlock()
	if bs != nil {
		bs.cancel()
	}
}

func (d *Dispatcher) runFinalize(batchID string, now time.Time) {
	if d.finalize == nil {
		return
	}
	if err := d.finalize(batchID, now); err != nil {
		d.logger.Error("batch lane: finalize failed", "batch_id", batchID, "error", err)
	}
}

// batchStateFor returns (creating if needed) the per-batch escalation and
// cancellation state. The context is derived from the tick's context, so a
// coordinator shutdown cancels every batch as well.
func (d *Dispatcher) batchStateFor(ctx context.Context, batchID string) *batchState {
	d.mu.Lock()
	defer d.mu.Unlock()
	if bs, ok := d.batches[batchID]; ok {
		return bs
	}
	bctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	bs := &batchState{
		bucket: TokenBucket{Rate: FloorItemsPerSec, Capacity: 1},
		ctx:    bctx,
		cancel: cancel,
	}
	d.batches[batchID] = bs
	return bs
}

// pruneBatchState drops the state of batches that are no longer open and have
// nothing outstanding, so a long-lived coordinator does not accumulate one
// entry per batch it has ever seen. A batch leaving the open list is also the
// dispatcher's signal that finalize took it terminal, which is when its result
// blobs start their retention clock.
func (d *Dispatcher) pruneBatchState(open []*store.Batch, now time.Time) {
	stillOpen := make(map[string]struct{}, len(open))
	for _, b := range open {
		stillOpen[b.ID] = struct{}{}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	for id, bs := range d.batches {
		if _, ok := stillOpen[id]; ok || bs.inflight > 0 {
			continue
		}
		bs.cancel()
		delete(d.batches, id)
		if _, scheduled := d.retire[id]; !scheduled {
			d.retire[id] = now.Add(d.cfg.OutputRetention)
		}
	}
}

// retention deletes the per-item result blobs of batches that finalized more
// than OutputRetention ago, and runs the assembled-file retention pass.
//
// The results are redundant by then: finalize inlines every one of them into
// the output file, whose own blob and row the Purge hook removes on the same
// boundary. Only what the dispatcher itself finalized is on the schedule, so a
// coordinator restart forgets the pending deletions and leaves those result
// blobs on disk (the assembled files still expire, because their rows carry the
// timestamp). Making that restart-safe needs a store read for terminal batches
// past a cutoff, which is a store change and is tracked as a follow-up.
func (d *Dispatcher) retention(now time.Time) {
	d.mu.Lock()
	due := make([]string, 0, len(d.retire))
	for id, at := range d.retire {
		if !now.Before(at) {
			due = append(due, id)
			delete(d.retire, id)
		}
	}
	runPurge := d.cfg.Purge != nil && (d.lastPurge.IsZero() || !now.Before(d.lastPurge.Add(d.cfg.PurgeInterval)))
	if runPurge {
		d.lastPurge = now
	}
	// The orphan pass never runs on the very first tick: a coordinator that has
	// just started has a cold store, a cold page cache and restart recovery
	// still settling, and the condition the pass repairs (a crash between
	// sealing an item body and committing its rows) has waited hours already. It
	// can wait one OrphanInterval more. lastOrphan is therefore seeded with the
	// first `now` the dispatcher sees rather than left zero.
	runOrphan := false
	switch {
	case d.lastOrphan.IsZero():
		d.lastOrphan = now
	case !now.Before(d.lastOrphan.Add(d.cfg.OrphanInterval)) && !d.orphanRunning:
		d.lastOrphan = now
		d.orphanRunning = true
		runOrphan = true
	}
	d.mu.Unlock()

	sort.Strings(due)
	for _, batchID := range due {
		d.purgeItemResults(batchID)
	}
	if runPurge {
		if _, err := d.cfg.Purge(now); err != nil {
			d.logger.Error("batch lane: file retention pass failed", "error", err)
		}
	}
	if runOrphan {
		// A full directory listing plus a store probe per candidate is the most
		// expensive thing the dispatcher does, and Tick is the 1 Hz control
		// loop: run it off the tick so a slow disk or a slow store cannot delay
		// a claim. It joins d.wg, so shutdown still waits for it.
		d.wg.Add(1)
		go func() {
			defer d.wg.Done()
			defer func() {
				d.mu.Lock()
				d.orphanRunning = false
				d.mu.Unlock()
			}()
			defer saferun.Recover(d.logger, "batch_orphan_sweep")
			d.sweepOrphanItemBlobs(now)
		}()
	}
}

// sweepOrphanItemBlobs deletes item blobs no row references.
//
// A coordinator that crashes between sealing an item body and committing the
// batch's rows leaves blobs behind that nothing will ever read or delete: the
// file retention pass walks file rows, and every other deletion path starts
// from an item row that does not exist. Only a directory listing can find them.
//
// Two guards keep the pass from deleting live data. It probes the store for
// each ref's item id, so anything a row still references is kept whatever its
// age; and it ignores blobs younger than the retention window, so a batch being
// created right now — its blobs written, its rows not yet committed — is never
// raced. Each pass is bounded on BOTH the probes it makes and the blobs it
// unlinks; a backlog drains over several. It runs off the tick, so neither
// bound is load bearing for the control loop's period.
func (d *Dispatcher) sweepOrphanItemBlobs(now time.Time) {
	blobs, err := d.blob.List()
	if err != nil {
		d.logger.Error("batch lane: could not list blobs for the orphan sweep", "error", err)
		return
	}
	cutoff := now.Add(-d.cfg.OutputRetention)

	scanned, deleted := 0, 0
	for _, info := range blobs {
		if deleted >= maxOrphanDeletes || scanned >= maxOrphanScan {
			d.logger.Info("batch lane: orphan sweep hit its per-pass bound",
				"deleted", deleted, "scanned", scanned)
			break
		}
		if !strings.HasPrefix(info.Ref, itemBlobPrefix) || !info.ModTime.Before(cutoff) {
			continue
		}
		scanned++
		itemID := strings.TrimSuffix(info.Ref, itemInputBlobSuffix)
		exists, err := d.st.BatchItemExists(itemID)
		if err != nil {
			d.logger.Error("batch lane: orphan sweep could not probe an item", "item_id", itemID, "error", err)
			return // a failing store read must not be read as "no row exists"
		}
		if exists {
			continue
		}
		if err := d.blob.Delete(info.Ref); err != nil && !errors.Is(err, sealedblob.ErrNotFound) {
			d.logger.Error("batch lane: could not delete an orphan blob", "item_id", itemID, "error", err)
			continue
		}
		deleted++
	}
	if deleted > 0 {
		d.logger.Info("batch lane: orphan sweep removed unreferenced item blobs",
			"blobs", deleted, "candidates", scanned)
	}
}

// purgeItemResults deletes every result blob of one finalized batch.
func (d *Dispatcher) purgeItemResults(batchID string) {
	items, err := d.st.ListItems(batchID)
	if err != nil {
		d.logger.Error("batch lane: could not list items for retention", "batch_id", batchID, "error", err)
		return
	}
	deleted := 0
	for _, it := range items {
		ref := it.ResultBlobRef
		if ref == "" {
			continue
		}
		if err := d.blob.Delete(ref); err != nil && !errors.Is(err, sealedblob.ErrNotFound) {
			d.logger.Error("batch lane: could not purge a result blob",
				"batch_id", batchID, "item_id", it.ID, "error", err)
			continue
		}
		deleted++
	}
	if deleted > 0 {
		d.logger.Info("batch lane: retention purged item results", "batch_id", batchID, "blobs", deleted)
	}
}

// SlotTarget reports one slot's current AIMD target. For tests and operational
// introspection.
func (d *Dispatcher) SlotTarget(key SlotKey) int {
	d.mu.Lock()
	defer d.mu.Unlock()
	if st, ok := d.slots[key]; ok {
		return st.aimd.Target
	}
	return 0
}

// InflightItems reports how many items the dispatcher currently has out.
func (d *Dispatcher) InflightItems() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.inflight
}
