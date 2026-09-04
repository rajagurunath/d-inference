package batchlane

// export_test.go opens the few internals the dispatcher's own tests need to the
// package's EXTERNAL test binary (package batchlane_test). Those tests are
// external because they drive the dispatcher with the doubles in
// batchlane/batchlanetest, and that package imports batchlane — an in-package
// test file importing it would be an import cycle.
//
// Nothing here is in the production build.

// AwaitDispatch waits for every dispatch goroutine and off-tick sweep the
// dispatcher has started so far, which is what makes a tick's asynchronous half
// deterministic in a test.
func (d *Dispatcher) AwaitDispatch() { d.wg.Wait() }

// The orphan sweep's per-pass bounds, so a test can assert against the real
// numbers rather than restate them.
const (
	MaxOrphanScan    = maxOrphanScan
	MaxOrphanDeletes = maxOrphanDeletes
)
