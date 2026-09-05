package batchlane

// settle.go is the settle half of the control loop: the drain step that starts
// every tick, the per-outcome accounting it feeds, and the sealing of a result
// blob. Split out of dispatcher.go, which keeps the loop itself — observe,
// rank, claim, dispatch.
//
// Privacy: nothing here logs a request body, a result, a custom_id or a
// metadata value. Log fields are ids, counts and bounded error codes.

import (
	"encoding/base64"
	"errors"
	"fmt"
	"time"

	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

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
