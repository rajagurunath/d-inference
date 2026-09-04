package api

import (
	"testing"

	"github.com/eigeninference/d-inference/coordinator/batchlane"
)

// The batch dispatcher writes an item's sealed result under
// batchlane.ResultBlobRef and the assembler reads it back under
// BatchItemResultRef. batchlane cannot import api (api imports batchlane), so
// the two derive the same ref independently — this pins them together, because
// a silent divergence would make every output file report a missing result.
func TestBatchItemResultRefMatchesTheDispatcher(t *testing.T) {
	for _, itemID := range []string{
		"bitem_0123456789abcdef01234567",
		"bitem_ffffffffffffffffffffffff",
	} {
		if got, want := batchlane.ResultBlobRef(itemID), BatchItemResultRef(itemID); got != want {
			t.Fatalf("batchlane.ResultBlobRef(%q) = %q, api.BatchItemResultRef = %q", itemID, got, want)
		}
	}
}

// The input ref must stay distinct from the result ref: finalize deletes every
// input blob the moment a batch settles, and a shared ref would take the
// results with it.
func TestBatchItemInputRefIsDistinctFromTheResultRef(t *testing.T) {
	const itemID = "bitem_0123456789abcdef01234567"
	if BatchItemInputRef(itemID) == BatchItemResultRef(itemID) {
		t.Fatal("the input and result blob refs collide")
	}
}
