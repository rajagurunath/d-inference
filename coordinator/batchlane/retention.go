package batchlane

// retention.go is the dispatcher's disk hygiene: the per-tick retention pass
// over finalized batches' result blobs, the hourly orphan sweep over blobs no
// row references, and the per-batch purge both of them drive. Split out of
// dispatcher.go, which keeps the control loop.
//
// Privacy: nothing here logs anything but ids and counts.

import (
	"errors"
	"sort"
	"strings"
	"time"

	"github.com/eigeninference/d-inference/coordinator/saferun"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

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
