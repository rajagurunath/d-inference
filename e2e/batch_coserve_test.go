package e2e

// batch_coserve_test.go — the Tidal co-serving benchmark (plan Task PR5).
//
// One suite, one provider, one model, four phases driven off ONE seeded
// arrival schedule so the online arms are comparable request for request:
//
//	online_only   Poisson ~0.5 req/s for 120 s, short prompts, nothing else running
//	offline_only  a 300-item batch alone; items/s over the middle 60 s is the
//	              offline ceiling
//	coserve       the same batch and the same online schedule together
//	flex          the same online schedule with service_tier: "batch" on every
//	              request (the OpenRouter synchronous path)
//
// Gates: coserve p99 / online_only p99 < 2.0, and harvest (coserve items/s ÷
// offline ceiling) > 0.2. The gates are floors; the report carries the real
// numbers, which is the point of the test.
//
// Set COSERVE_REPORT_PATH to write the rendered markdown report; without it the
// report is only logged.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/e2e/testbed"
)

const (
	// coServeSeed fixes the arrival schedule so online_only, coserve and flex
	// replay the same inter-arrival times. Changing it invalidates comparison
	// with any previously published report.
	coServeSeed = int64(20260906)
	// coServeOnlineRate is the Poisson arrival rate of the online arm.
	coServeOnlineRate = 0.5
	// coServeOnlineWindow is how long the schedule runs.
	coServeOnlineWindow = 120 * time.Second
	// coServeBatchItems is the size of the offline job in each batch phase.
	coServeBatchItems = 300
	// coServeBatchMaxTokens / coServeOnlineMaxTokens keep both arms short so a
	// phase is bounded by the schedule rather than by generation length.
	coServeBatchMaxTokens  = 24
	coServeOnlineMaxTokens = 32
	// coServeWarmup is skipped before the throughput window opens: the first
	// seconds of a batch are dispatcher ramp (AIMD opening the per-slot cap),
	// not steady state.
	coServeWarmup = 30 * time.Second
	// coServeMeasure is the width of the throughput window.
	coServeMeasure = 60 * time.Second
	// coServePoll is the batch progress sampling interval.
	coServePoll = time.Second
	// coServeQuiet lets a cancelled batch's in-flight items drain before the
	// next phase starts measuring.
	coServeQuiet = 15 * time.Second

	// Gates from the plan.
	coServeP99RatioGate = 2.0
	coServeHarvestGate  = 0.2
)

// ---------------------------------------------------------------- samples

// onlineSample is one online request's outcome. Duration is client-observed
// wall time: the coordinator emits a completion as a single SSE frame, so TTFT
// is not separable here (see the report's caveats).
type onlineSample struct {
	Index    int
	Offset   time.Duration
	Duration time.Duration
	Status   int
	Err      error
}

// onlineStats summarises one online arm.
type onlineStats struct {
	Total   int
	OK      int
	Reject  int // HTTP 429
	Other   int // any other non-200, transport errors included
	P50     time.Duration
	P99     time.Duration
	Mean    time.Duration
	Max     time.Duration
	Elapsed time.Duration
}

func summariseOnline(samples []onlineSample, elapsed time.Duration) onlineStats {
	st := onlineStats{Total: len(samples), Elapsed: elapsed}
	var ok []time.Duration
	var total time.Duration
	for _, s := range samples {
		switch {
		case s.Status == http.StatusOK:
			st.OK++
			ok = append(ok, s.Duration)
			total += s.Duration
		case s.Status == http.StatusTooManyRequests:
			st.Reject++
		default:
			st.Other++
		}
	}
	if len(ok) == 0 {
		return st
	}
	sort.Slice(ok, func(i, j int) bool { return ok[i] < ok[j] })
	st.P50 = nearestRank(ok, 50)
	st.P99 = nearestRank(ok, 99)
	st.Max = ok[len(ok)-1]
	st.Mean = total / time.Duration(len(ok))
	return st
}

// nearestRank matches the percentile convention testbed.LoadResult already
// uses (sorted[len*p/100], clamped), so numbers here and in the load benchmarks
// mean the same thing.
func nearestRank(sorted []time.Duration, pct int) time.Duration {
	idx := len(sorted) * pct / 100
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// batchSample is one poll of a running batch, timestamped from the moment the
// batch was created.
type batchSample struct {
	At        time.Duration
	Completed int
	Failed    int
	Status    string
}

// batchRate is items/s measured across a window of batch samples.
type batchRate struct {
	ItemsPerSec float64
	From, To    time.Duration
	FromItems   int
	ToItems     int
	Samples     int
}

// batchRateOverWindow measures settled items per second between the first
// sample at or after from and the last sample at or before to. It returns
// ok=false when the phase produced no usable window (batch finished before the
// window opened, or fewer than two samples landed inside it).
func batchRateOverWindow(samples []batchSample, from, to time.Duration) (batchRate, bool) {
	var lo, hi *batchSample
	for i := range samples {
		s := &samples[i]
		if s.At >= from && lo == nil {
			lo = s
		}
		if s.At <= to {
			hi = s
		}
	}
	if lo == nil || hi == nil || hi.At <= lo.At {
		return batchRate{}, false
	}
	settled := func(s *batchSample) int { return s.Completed + s.Failed }
	span := (hi.At - lo.At).Seconds()
	return batchRate{
		ItemsPerSec: float64(settled(hi)-settled(lo)) / span,
		From:        lo.At,
		To:          hi.At,
		FromItems:   settled(lo),
		ToItems:     settled(hi),
		Samples:     len(samples),
	}, true
}

// ---------------------------------------------------------------- harness

type coServeHarness struct {
	t        *testing.T
	suite    *testbed.Suite
	batch    *testbed.BatchClient
	apiKey   string
	model    string
	baseURL  string
	client   *http.Client
	schedule []time.Duration
}

// poissonSchedule generates seeded Poisson arrival offsets inside window.
func poissonSchedule(seed int64, rate float64, window time.Duration) []time.Duration {
	rng := rand.New(rand.NewSource(seed))
	var (
		offsets []time.Duration
		elapsed float64
	)
	for {
		elapsed += -math.Log(1-rng.Float64()) / rate
		offset := time.Duration(elapsed * float64(time.Second))
		if offset >= window {
			return offsets
		}
		offsets = append(offsets, offset)
	}
}

// onlineRequest issues one synchronous chat completion and times it.
// serviceTier "" is an ordinary online request; "batch" is the OpenRouter
// synchronous batch-lane path.
func (h *coServeHarness) onlineRequest(ctx context.Context, idx int, serviceTier string) onlineSample {
	body := map[string]any{
		"model":       h.model,
		"messages":    []map[string]string{{"role": "user", "content": fmt.Sprintf("What is %d+%d? Answer with just the number.", idx, idx+1)}},
		"stream":      false,
		"max_tokens":  coServeOnlineMaxTokens,
		"temperature": 0.0,
	}
	if serviceTier != "" {
		body["service_tier"] = serviceTier
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return onlineSample{Index: idx, Err: err}
	}

	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		h.baseURL+"/v1/chat/completions", strings.NewReader(string(payload)))
	if err != nil {
		return onlineSample{Index: idx, Err: err, Duration: time.Since(start)}
	}
	req.Header.Set("Authorization", "Bearer "+h.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := h.client.Do(req)
	if err != nil {
		return onlineSample{Index: idx, Err: err, Duration: time.Since(start)}
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	return onlineSample{Index: idx, Status: resp.StatusCode, Duration: time.Since(start)}
}

// runSchedule replays the seeded arrival schedule, firing each request at its
// own offset regardless of how long the previous one took (an open-loop
// arrival process, not a closed loop that self-throttles under load).
func (h *coServeHarness) runSchedule(ctx context.Context, serviceTier string) ([]onlineSample, time.Duration) {
	samples := make([]onlineSample, len(h.schedule))
	start := time.Now()

	var wg sync.WaitGroup
	for i, offset := range h.schedule {
		wg.Add(1)
		go func(i int, offset time.Duration) {
			defer wg.Done()
			timer := time.NewTimer(time.Until(start.Add(offset)))
			defer timer.Stop()
			select {
			case <-ctx.Done():
				samples[i] = onlineSample{Index: i, Offset: offset, Err: ctx.Err()}
				return
			case <-timer.C:
			}
			s := h.onlineRequest(ctx, i, serviceTier)
			s.Offset = offset
			samples[i] = s
		}(i, offset)
	}
	wg.Wait()
	return samples, time.Since(start)
}

// batchLines builds a coServeBatchItems-line offline job. The prompts are
// synthetic arithmetic, the same family the online arm uses, so the two lanes
// contend for the same kind of work.
func (h *coServeHarness) batchLines(prefix string, n int) []testbed.BatchInputLine {
	lines := make([]testbed.BatchInputLine, n)
	for i := range lines {
		lines[i] = testbed.BatchInputLine{
			CustomID:  fmt.Sprintf("%s-%d", prefix, i),
			Model:     h.model,
			Prompt:    fmt.Sprintf("What is %d+%d? Answer with just the number.", i, i+7),
			MaxTokens: coServeBatchMaxTokens,
		}
	}
	return lines
}

// runBatch submits an offline job and samples its progress once a second until
// `hold` elapses or it settles, then cancels whatever is left so the next phase
// starts from a quiet stack.
//
// It returns an error rather than calling require: the co-serving phase runs it
// on its own goroutine, and testify's FailNow there would only Goexit that
// goroutine.
func (h *coServeHarness) runBatch(ctx context.Context, prefix string, hold time.Duration) ([]batchSample, error) {
	t := h.t

	batch, err := h.batch.SubmitBatch(ctx, prefix+".jsonl", h.batchLines(prefix, coServeBatchItems))
	if err != nil {
		return nil, fmt.Errorf("submit %s batch: %w", prefix, err)
	}
	if batch.RequestCounts.Total != coServeBatchItems {
		return nil, fmt.Errorf("%s batch admitted %d of %d items",
			prefix, batch.RequestCounts.Total, coServeBatchItems)
	}
	t.Logf("[%s] batch %s admitted with %d items", prefix, batch.ID, batch.RequestCounts.Total)

	start := time.Now()
	var samples []batchSample
	pollCtx, cancel := context.WithTimeout(ctx, hold+coServePoll)
	defer cancel()

	last, err := h.batch.PollWhile(pollCtx, batch.ID, coServePoll, func(b testbed.BatchObject) bool {
		samples = append(samples, batchSample{
			At:        time.Since(start),
			Completed: b.RequestCounts.Completed,
			Failed:    b.RequestCounts.Failed,
			Status:    b.Status,
		})
		return time.Since(start) < hold
	})
	if err != nil && ctx.Err() == nil {
		t.Logf("[%s] batch polling ended early: %v", prefix, err)
	}
	t.Logf("[%s] batch %s after %s: status=%s completed=%d failed=%d",
		prefix, batch.ID, time.Since(start).Round(time.Second),
		last.Status, last.RequestCounts.Completed, last.RequestCounts.Failed)

	if !testbed.BatchIsTerminal(last.Status) {
		if _, err := h.batch.CancelBatch(context.WithoutCancel(ctx), batch.ID); err != nil {
			t.Logf("[%s] cancelling batch %s failed: %v", prefix, batch.ID, err)
		}
	}
	return samples, nil
}

// ---------------------------------------------------------------- earnings

// laneEarnings is the provider's payout for one lane over one phase.
type laneEarnings struct {
	Rows     int
	MicroUSD int64
}

// earningsByLane buckets provider_earnings rows created inside [from, to) by
// their Lane column ("" is the online lane, "batch" the discounted one).
func earningsByLane(t *testing.T, st store.Store, accountID string, from, to time.Time) map[string]laneEarnings {
	t.Helper()
	out := map[string]laneEarnings{}
	if accountID == "" {
		return out
	}
	rows, err := st.GetAccountEarnings(accountID, 1_000_000)
	if err != nil {
		t.Logf("reading provider earnings failed: %v", err)
		return out
	}
	for _, row := range rows {
		if row.Model == "base_reward" {
			continue
		}
		if row.CreatedAt.Before(from) || !row.CreatedAt.Before(to) {
			continue
		}
		lane := row.Lane
		if lane == "" {
			lane = "online"
		}
		agg := out[lane]
		agg.Rows++
		agg.MicroUSD += row.AmountMicroUSD
		out[lane] = agg
	}
	return out
}

// earningsPerHour scales a phase's payout to a one-hour rate, in USD.
func earningsPerHour(micro int64, phase time.Duration) float64 {
	if phase <= 0 {
		return 0
	}
	return float64(micro) / 1e6 / phase.Seconds() * 3600
}

// providerAccountID returns the account the suite's provider is paid into.
func providerAccountID(s *testbed.Suite) string {
	var accountID string
	s.Coordinator.Registry.ForEachProvider(func(p *registry.Provider) {
		p.Mu().Lock()
		defer p.Mu().Unlock()
		if accountID == "" {
			accountID = p.AccountID
		}
	})
	return accountID
}

// ---------------------------------------------------------------- the test

func TestBenchmarkBatchCoServe(t *testing.T) {
	if os.Getenv("DARKBLOOM_PROVIDER_BINARY") == "" {
		t.Skip("set DARKBLOOM_PROVIDER_BINARY (and a local MLX checkpoint) to run the co-serving benchmark")
	}
	// The batch lane is opt-in: without a key it is not wired at all and every
	// batch route answers 503. A benchmark run is local by definition, so
	// default the dev escape hatch on rather than failing on a missing export.
	if os.Getenv("EIGENINFERENCE_BATCH_DEV_INSECURE_KEY") == "" {
		t.Setenv("EIGENINFERENCE_BATCH_DEV_INSECURE_KEY", "true")
	}
	if os.Getenv("EIGENINFERENCE_BATCH_BLOB_DIR") == "" {
		t.Setenv("EIGENINFERENCE_BATCH_BLOB_DIR", t.TempDir())
	}

	ctx := context.Background()
	suiteCfg := testbed.SuiteConfig{
		ModelSpecs: []testbed.ModelSpec{{ModelID: testbed.DefaultTestModelID(), NumProviders: 1}},
		// Two accounts on purpose: the provider is owned by testbed-user-0 and
		// every request here is issued by testbed-user-1, so the provider's
		// earnings rows are unambiguously payouts for someone else's traffic.
		NumUsers:       2,
		QueueCapacity:  200,
		QueueTimeout:   120 * time.Second,
		SeedBalance:    2_000_000_000,
		UseMemoryStore: true,
	}

	s := testbed.NewSuite(suiteCfg)
	require.NoError(t, s.Start(ctx), "suite startup failed")
	t.Cleanup(s.Stop)
	require.NotNil(t, s.Coordinator.Server.BatchBlobs(),
		"batch lane is not wired — set EIGENINFERENCE_BATCH_DEV_INSECURE_KEY and EIGENINFERENCE_BATCH_BLOB_DIR")
	require.GreaterOrEqual(t, len(s.Users), 2)

	consumer := s.Users[1]
	h := &coServeHarness{
		t:        t,
		suite:    s,
		batch:    testbed.NewBatchClient(s, consumer.APIKey),
		apiKey:   consumer.APIKey,
		model:    s.PrimaryModelID(),
		baseURL:  s.Coordinator.BaseURL(),
		client:   &http.Client{Timeout: httpTimeout},
		schedule: poissonSchedule(coServeSeed, coServeOnlineRate, coServeOnlineWindow),
	}
	require.NotEmpty(t, h.schedule, "seeded arrival schedule must not be empty")
	t.Logf("seeded arrival schedule: %d requests over %s (seed %d, rate %.2f req/s)",
		len(h.schedule), coServeOnlineWindow, coServeSeed, coServeOnlineRate)

	// Warm the model so the first measured request does not pay for a lazy
	// load. The benchmark measures steady co-serving, not cold start.
	for i := 0; i < 2; i++ {
		warm := h.onlineRequest(ctx, -1-i, "")
		require.Equal(t, http.StatusOK, warm.Status,
			"model warm-up request %d failed (err=%v)", i, warm.Err)
		t.Logf("warm-up %d: %s", i, warm.Duration.Round(time.Millisecond))
	}

	var (
		onlineOnly   onlineStats
		coserve      onlineStats
		flex         onlineStats
		ceiling      batchRate
		harvestRate  batchRate
		phaseWindows = map[string][2]time.Time{}
		phaseElapsed = map[string]time.Duration{}
	)

	record := func(name string, start time.Time, elapsed time.Duration) {
		phaseWindows[name] = [2]time.Time{start, start.Add(elapsed)}
		phaseElapsed[name] = elapsed
	}

	ok := t.Run("online_only", func(t *testing.T) {
		start := time.Now()
		samples, elapsed := h.runSchedule(ctx, "")
		record("online_only", start, elapsed)
		onlineOnly = summariseOnline(samples, elapsed)
		t.Logf("online_only: n=%d ok=%d 429=%d other=%d p50=%s p99=%s mean=%s max=%s",
			onlineOnly.Total, onlineOnly.OK, onlineOnly.Reject, onlineOnly.Other,
			onlineOnly.P50.Round(time.Millisecond), onlineOnly.P99.Round(time.Millisecond),
			onlineOnly.Mean.Round(time.Millisecond), onlineOnly.Max.Round(time.Millisecond))
		require.Greater(t, onlineOnly.OK, 0, "the online-only baseline served nothing")
	})
	require.True(t, ok, "online_only phase failed; later phases have no baseline")

	ok = t.Run("offline_only", func(t *testing.T) {
		start := time.Now()
		samples, err := h.runBatch(ctx, "offline", coServeWarmup+coServeMeasure)
		record("offline_only", start, time.Since(start))
		require.NoError(t, err)
		var found bool
		ceiling, found = batchRateOverWindow(samples, coServeWarmup, coServeWarmup+coServeMeasure)
		require.True(t, found,
			"no usable throughput window in %d samples — the batch settled before %s",
			len(samples), coServeWarmup)
		t.Logf("offline ceiling: %.3f items/s over [%s, %s] (%d -> %d items)",
			ceiling.ItemsPerSec, ceiling.From.Round(time.Second), ceiling.To.Round(time.Second),
			ceiling.FromItems, ceiling.ToItems)
		require.Greater(t, ceiling.ItemsPerSec, 0.0, "the batch made no progress alone")
	})
	require.True(t, ok, "offline_only phase failed; harvest has no denominator")

	quiesce(t, coServeQuiet)

	ok = t.Run("coserve", func(t *testing.T) {
		hold := coServeWarmup + coServeMeasure
		if coServeOnlineWindow > hold {
			hold = coServeOnlineWindow
		}
		start := time.Now()

		var (
			wg            sync.WaitGroup
			batchSamples  []batchSample
			batchErr      error
			onlineSamples []onlineSample
			onlineElapsed time.Duration
		)
		wg.Add(2)
		go func() {
			defer wg.Done()
			batchSamples, batchErr = h.runBatch(ctx, "coserve", hold)
		}()
		go func() {
			defer wg.Done()
			onlineSamples, onlineElapsed = h.runSchedule(ctx, "")
		}()
		wg.Wait()
		record("coserve", start, time.Since(start))
		require.NoError(t, batchErr)

		coserve = summariseOnline(onlineSamples, onlineElapsed)
		var found bool
		harvestRate, found = batchRateOverWindow(batchSamples, coServeWarmup, coServeWarmup+coServeMeasure)
		require.True(t, found, "no usable co-serving throughput window in %d samples", len(batchSamples))

		t.Logf("coserve online: n=%d ok=%d 429=%d other=%d p50=%s p99=%s mean=%s max=%s",
			coserve.Total, coserve.OK, coserve.Reject, coserve.Other,
			coserve.P50.Round(time.Millisecond), coserve.P99.Round(time.Millisecond),
			coserve.Mean.Round(time.Millisecond), coserve.Max.Round(time.Millisecond))
		t.Logf("coserve batch: %.3f items/s over [%s, %s] (%d -> %d items)",
			harvestRate.ItemsPerSec, harvestRate.From.Round(time.Second), harvestRate.To.Round(time.Second),
			harvestRate.FromItems, harvestRate.ToItems)
		require.Greater(t, coserve.OK, 0, "online traffic stopped entirely under batch load")
	})
	require.True(t, ok, "coserve phase failed")

	quiesce(t, coServeQuiet)

	ok = t.Run("flex", func(t *testing.T) {
		start := time.Now()
		samples, elapsed := h.runSchedule(ctx, "batch")
		record("flex", start, elapsed)
		flex = summariseOnline(samples, elapsed)
		t.Logf("flex (service_tier=batch): n=%d 200=%d 429=%d other=%d p50=%s p99=%s mean=%s max=%s",
			flex.Total, flex.OK, flex.Reject, flex.Other,
			flex.P50.Round(time.Millisecond), flex.P99.Round(time.Millisecond),
			flex.Mean.Round(time.Millisecond), flex.Max.Round(time.Millisecond))
		// A service_tier=batch request is expected to be served or refused with
		// a 429 carrying Retry-After. Anything else is recorded rather than
		// asserted: the report's job is to say what happened, and only the two
		// plan gates below fail the run.
		if flex.Other > 0 {
			t.Logf("WARNING: %d service_tier=batch requests ended on neither 200 nor 429", flex.Other)
		}
		require.Greater(t, flex.OK+flex.Reject, 0, "the flex arm produced no admission decision at all")
	})
	require.True(t, ok, "flex phase failed")

	// --- gates -----------------------------------------------------------

	p99Ratio := ratio(coserve.P99, onlineOnly.P99)
	p50Ratio := ratio(coserve.P50, onlineOnly.P50)
	harvest := 0.0
	if ceiling.ItemsPerSec > 0 {
		harvest = harvestRate.ItemsPerSec / ceiling.ItemsPerSec
	}

	t.Logf("GATES: p99 ratio %.2f (< %.1f), harvest %.2f (> %.2f)",
		p99Ratio, coServeP99RatioGate, harvest, coServeHarvestGate)

	// --- earnings --------------------------------------------------------

	payoutAccount := providerAccountID(s)
	earnings := map[string]map[string]laneEarnings{}
	for _, phase := range []string{"online_only", "offline_only", "coserve", "flex"} {
		window := phaseWindows[phase]
		earnings[phase] = earningsByLane(t, s.PgStore, payoutAccount, window[0], window[1])
		for lane, agg := range earnings[phase] {
			t.Logf("earnings %s/%s: %d rows, %.4f USD -> %.4f USD/h",
				phase, lane, agg.Rows, float64(agg.MicroUSD)/1e6,
				earningsPerHour(agg.MicroUSD, phaseElapsed[phase]))
		}
	}

	// --- report ----------------------------------------------------------

	report := buildCoServeReport(coServeReportInput{
		Suite:        s,
		Model:        h.model,
		Schedule:     h.schedule,
		OnlineOnly:   onlineOnly,
		CoServe:      coserve,
		Flex:         flex,
		Ceiling:      ceiling,
		HarvestRate:  harvestRate,
		Harvest:      harvest,
		P50Ratio:     p50Ratio,
		P99Ratio:     p99Ratio,
		Earnings:     earnings,
		PhaseElapsed: phaseElapsed,
	})
	rendered := report.Render()
	t.Logf("\n%s", rendered)
	if path := os.Getenv("COSERVE_REPORT_PATH"); path != "" {
		require.NoError(t, os.WriteFile(path, []byte(rendered), 0o644))
		t.Logf("report written to %s", path)
	}

	require.Less(t, p99Ratio, coServeP99RatioGate,
		"co-serving online p99 regressed more than %.1fx against the in-session baseline",
		coServeP99RatioGate)
	require.Greater(t, harvest, coServeHarvestGate,
		"co-serving harvested less than %.0f%% of the offline ceiling",
		coServeHarvestGate*100)
}

// quiesce waits for in-flight work from the previous phase to drain.
func quiesce(t *testing.T, d time.Duration) {
	t.Helper()
	t.Logf("quiescing for %s before the next phase", d)
	time.Sleep(d)
}

func ratio(measured, baseline time.Duration) float64 {
	if baseline <= 0 {
		return 0
	}
	return float64(measured) / float64(baseline)
}

// ---------------------------------------------------------------- report

type coServeReportInput struct {
	Suite        *testbed.Suite
	Model        string
	Schedule     []time.Duration
	OnlineOnly   onlineStats
	CoServe      onlineStats
	Flex         onlineStats
	Ceiling      batchRate
	HarvestRate  batchRate
	Harvest      float64
	P50Ratio     float64
	P99Ratio     float64
	Earnings     map[string]map[string]laneEarnings
	PhaseElapsed map[string]time.Duration
}

func buildCoServeReport(in coServeReportInput) testbed.CoServeReport {
	ms := func(d time.Duration) string { return fmt.Sprintf("%d ms", d.Round(time.Millisecond).Milliseconds()) }

	metrics := []testbed.CoServeMetric{
		{
			Name:       "offline ceiling",
			Definition: fmt.Sprintf("batch items settled per second with no online load, measured over the %s window that opens %s after the batch is created", coServeMeasure, coServeWarmup),
			Value:      fmt.Sprintf("%.3f items/s (%d → %d items over %s)", in.Ceiling.ItemsPerSec, in.Ceiling.FromItems, in.Ceiling.ToItems, (in.Ceiling.To - in.Ceiling.From).Round(time.Second)),
		},
		{
			Name:       "co-serving batch rate",
			Definition: "the same measurement while the online arrival schedule replays against the same provider",
			Value:      fmt.Sprintf("%.3f items/s (%d → %d items over %s)", in.HarvestRate.ItemsPerSec, in.HarvestRate.FromItems, in.HarvestRate.ToItems, (in.HarvestRate.To - in.HarvestRate.From).Round(time.Second)),
		},
		{
			Name:       "harvest",
			Definition: "co-serving batch rate ÷ offline ceiling — the fraction of the idle-capacity ceiling still reachable while online traffic is served",
			Value:      fmt.Sprintf("%.2f (%.0f%%)", in.Harvest, in.Harvest*100),
		},
		{
			Name:       "online-only p50 / p99",
			Definition: "client-observed wall time of the seeded Poisson arrivals with nothing else running",
			Value:      fmt.Sprintf("%s / %s (n=%d served)", ms(in.OnlineOnly.P50), ms(in.OnlineOnly.P99), in.OnlineOnly.OK),
		},
		{
			Name:       "co-serving p50 / p99",
			Definition: "the same schedule, same seed, while the offline job runs",
			Value:      fmt.Sprintf("%s / %s (n=%d served)", ms(in.CoServe.P50), ms(in.CoServe.P99), in.CoServe.OK),
		},
		{
			Name:       "online p50 ratio",
			Definition: "co-serving p50 ÷ online-only p50",
			Value:      fmt.Sprintf("%.2f×", in.P50Ratio),
		},
		{
			Name:       "online p99 ratio",
			Definition: "co-serving p99 ÷ online-only p99 — the gated number",
			Value:      fmt.Sprintf("%.2f×", in.P99Ratio),
		},
		{
			Name:       "flex admission",
			Definition: `the same schedule with service_tier: "batch" on every synchronous request (the OpenRouter path): HTTP 200 vs 429`,
			Value:      fmt.Sprintf("%d × 200, %d × 429 (of %d)", in.Flex.OK, in.Flex.Reject, in.Flex.Total),
		},
		{
			Name:       "flex p50 / p99",
			Definition: "wall time of the served service_tier=batch requests",
			Value:      fmt.Sprintf("%s / %s", ms(in.Flex.P50), ms(in.Flex.P99)),
		},
	}

	for _, phase := range []string{"online_only", "offline_only", "coserve", "flex"} {
		lanes := in.Earnings[phase]
		elapsed := in.PhaseElapsed[phase]
		metrics = append(metrics, testbed.CoServeMetric{
			Name:       "earnings/hour — " + phase,
			Definition: fmt.Sprintf("Σ provider_earnings rows created during the phase, by Lane, scaled from the phase's %s to one hour", elapsed.Round(time.Second)),
			Value:      formatLaneEarnings(lanes, elapsed),
		})
	}

	return testbed.CoServeReport{
		Title: "Batch co-serving benchmark",
		Intro: "What one Mac gives back when the Tidal batch lane fills the gaps between online requests, " +
			"and what that costs the online requests. Produced by `TestBenchmarkBatchCoServe` " +
			"(`e2e/batch_coserve_test.go`) against a real coordinator, a real Swift provider and real MLX inference.",
		Setup: []testbed.CoServeSetting{
			{Name: "host", Value: envOr("RUNNER_DESC", fmt.Sprintf("local %s/%s", runtime.GOOS, runtime.GOARCH))},
			{Name: "run", Value: time.Now().UTC().Format("2006-01-02 15:04 UTC")},
			{Name: "model", Value: "`" + in.Model + "`"},
			{Name: "providers", Value: fmt.Sprintf("%d", len(in.Suite.Providers))},
			{Name: "store", Value: "in-memory testbed store"},
			{Name: "arrival schedule", Value: fmt.Sprintf("seeded Poisson, seed %d, %.2f req/s, %s window, %d arrivals", coServeSeed, coServeOnlineRate, coServeOnlineWindow, len(in.Schedule))},
			{Name: "offline job", Value: fmt.Sprintf("%d items, max_tokens %d", coServeBatchItems, coServeBatchMaxTokens)},
			{Name: "online requests", Value: fmt.Sprintf("non-streaming, max_tokens %d, short arithmetic prompts", coServeOnlineMaxTokens)},
		},
		Method: []string{
			"All four phases share one suite, one provider process and one loaded model; the model is warmed with two throwaway requests before the first measurement.",
			"`online_only` — replay the seeded schedule with nothing else running. This is the baseline every ratio below is taken against, measured in the same session on the same box, never against a historical number.",
			fmt.Sprintf("`offline_only` — submit a %d-item batch alone, sample `GET /v1/batches/{id}` once a second, and take items/s over the %s window that opens %s in (the first seconds are the dispatcher's AIMD ramp, not steady state). The batch is cancelled once the window closes.", coServeBatchItems, coServeMeasure, coServeWarmup),
			"`coserve` — the same batch and the same arrival schedule together, measured the same way on both sides.",
			`flex — the same schedule again with service_tier: "batch" on every synchronous request, counting 200s against 429s.`,
			"Earnings come from the store's `provider_earnings` rows, bucketed by their `Lane` column over each phase's wall-clock window and scaled to an hourly rate.",
		},
		Gates: []string{
			fmt.Sprintf("co-serving p99 ÷ online-only p99 = **%.2f×**, gate < %.1f — %s", in.P99Ratio, coServeP99RatioGate, pass(in.P99Ratio < coServeP99RatioGate)),
			fmt.Sprintf("harvest = **%.2f**, gate > %.2f — %s", in.Harvest, coServeHarvestGate, pass(in.Harvest > coServeHarvestGate)),
		},
		Caveats: []string{
			"**One provider, one model, one box.** Nothing here says how the lane behaves across a fleet; it says the lane harvests idle capacity on a single machine without starving the online lane.",
			"**Debug provider build.** The Swift provider is built in debug with a hand-built `mlx.metallib`; absolute latencies and tokens/s are lower than a release build's.",
			"**TTFT is not observable.** The coordinator hashes and SE-signs a completion before releasing it, so it emits the whole response as one SSE frame. Every online number here is total wall time, not time-to-first-token.",
			fmt.Sprintf("**Small n.** The schedule produces %d arrivals per phase, so p99 is effectively the worst observed request rather than a stable tail estimate. The hand-run predecessor of this benchmark (`plans/e2e-batch-run.md` step 7, n=10 per arm) saw the online penalty land anywhere between 1.2× and 1.9× depending on the round; read a single ratio in that spread as consistent with the earlier measurement, not as a tighter result than the sample supports.", len(in.Schedule)),
			"**Phases are sequential, not simultaneous.** The baseline and the co-serving arm are minutes apart on the same box, so slow drift (thermals, other processes) is inside the ratio.",
			"**The offline job is cancelled, not drained.** Both batch phases stop once the measurement window closes; the ceiling and the co-serving rate are steady-state rates, not end-to-end completion times for the whole 300 items.",
		},
		Metrics: metrics,
	}
}

func formatLaneEarnings(lanes map[string]laneEarnings, elapsed time.Duration) string {
	if len(lanes) == 0 {
		return "no earnings rows"
	}
	names := make([]string, 0, len(lanes))
	for lane := range lanes {
		names = append(names, lane)
	}
	sort.Strings(names)
	parts := make([]string, 0, len(names))
	for _, lane := range names {
		agg := lanes[lane]
		parts = append(parts, fmt.Sprintf("%s $%.4f/h (%d rows)",
			lane, earningsPerHour(agg.MicroUSD, elapsed), agg.Rows))
	}
	return strings.Join(parts, ", ")
}

func pass(ok bool) string {
	if ok {
		return "**PASS**"
	}
	return "**FAIL**"
}
