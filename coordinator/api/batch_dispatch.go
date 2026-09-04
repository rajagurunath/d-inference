package api

// batch_dispatch.go is the api-layer surface of the batch lane
// (docs/design/tidal-batch-lane.md §3.4): the vocabulary the no-capacity
// terminal answers with, and the entry point the batch dispatcher uses to run
// one item through the ordinary dispatch funnel.

const (
	// batchNoCapacityCode is the bounded error/rejection code a batch-lane
	// request gets when no provider slot has headroom for it. It is also the
	// ErrCode BatchOutcome carries, so the batch dispatcher can tell "nothing
	// free right now, re-offer me next tick" apart from a real failure without
	// parsing prose.
	batchNoCapacityCode = "no_capacity"
	// batchNoCapacityRetryAfterSec is the Retry-After the batch lane advertises
	// on that 429. It matches the dispatcher's 1 Hz tick scale: a few seconds is
	// long enough for online traffic to drain a slot and short enough that a
	// paced client (OpenRouter's synchronous service_tier=batch calls) keeps
	// making progress.
	batchNoCapacityRetryAfterSec = 5
)
