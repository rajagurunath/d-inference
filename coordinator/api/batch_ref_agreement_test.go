package api

import (
	"testing"

	"github.com/eigeninference/d-inference/coordinator/batchlane"
)

// api now imports batchlane, so the dispatcher's Outcome, its result-blob ref
// and its retention window are used directly instead of being mirrored here.
// This is the compile-time pin that the entry point still IS the funnel type:
// a signature change on either side stops being a silent adapter edit in
// cmd/coordinator and becomes a build failure.
var _ batchlane.DispatchFn = (*Server)(nil).DispatchBatchItem

// The input ref is the one blob helper api still owns — batchlane exports no
// input-ref function because the dispatcher never derives one, it reads the ref
// off the item row. It must stay distinct from the result ref: finalize deletes
// every input blob the moment a batch settles, and a shared ref would take the
// results with it.
func TestBatchItemInputRefIsDistinctFromTheResultRef(t *testing.T) {
	const itemID = "bitem_0123456789abcdef01234567"
	if BatchItemInputRef(itemID) == batchlane.ResultBlobRef(itemID) {
		t.Fatal("the input and result blob refs collide")
	}
}
