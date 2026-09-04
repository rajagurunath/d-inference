// Package batchlanetest holds the test doubles for the batch dispatcher: a
// mutable registry view, a scriptable dispatch funnel and a recording finalize
// hook.
//
// They live in their own package rather than in batchlane so the production
// build of batchlane carries no fakes, while PR5's co-serving benchmark — which
// drives the dispatcher without a provider fleet — can still import them. That
// is also why batchlane's own dispatcher tests are an external test package
// (batchlane_test): an in-package test file may not import a package that
// imports batchlane.
package batchlanetest

import (
	"context"
	"sync"
	"time"

	"github.com/eigeninference/d-inference/coordinator/batchlane"
)

// FakeView is a RegistryView whose signals the test sets directly. Safe for
// concurrent use so a test can move a slot under a running dispatcher.
type FakeView struct {
	mu    sync.Mutex
	slots map[batchlane.SlotKey]batchlane.SlotSignal
}

// NewFakeView returns an empty view.
func NewFakeView() *FakeView { return &FakeView{slots: map[batchlane.SlotKey]batchlane.SlotSignal{}} }

// Set installs (or replaces) one slot's signal.
func (v *FakeView) Set(key batchlane.SlotKey, sig batchlane.SlotSignal) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.slots[key] = sig
}

// Update applies fn to one slot's signal in place.
func (v *FakeView) Update(key batchlane.SlotKey, fn func(*batchlane.SlotSignal)) {
	v.mu.Lock()
	defer v.mu.Unlock()
	sig := v.slots[key]
	fn(&sig)
	v.slots[key] = sig
}

// Remove drops a slot, as if the provider disconnected.
func (v *FakeView) Remove(key batchlane.SlotKey) {
	v.mu.Lock()
	defer v.mu.Unlock()
	delete(v.slots, key)
}

// Slots implements RegistryView.
func (v *FakeView) Slots(string) map[batchlane.SlotKey]batchlane.SlotSignal {
	v.mu.Lock()
	defer v.mu.Unlock()
	out := make(map[batchlane.SlotKey]batchlane.SlotSignal, len(v.slots))
	for k, s := range v.slots {
		out[k] = s
	}
	return out
}

// DispatchCall records one call into the fake funnel. The request body is
// deliberately NOT retained: the dispatcher must be able to run without any
// component holding prompt bytes past the call, and a fake that stored them
// would hide a leak rather than expose it.
type DispatchCall struct {
	AccountID string
	APIKeyID  string
	Model     string
	BodyLen   int
}

// FakeDispatch is a scriptable stand-in for (*api.Server).DispatchBatchItem.
// The zero value returns an empty success for every call.
type FakeDispatch struct {
	mu    sync.Mutex
	calls []DispatchCall
	// Respond returns the outcome for call n (zero-based). Nil means a bare
	// success.
	Respond func(n int, model string, body []byte) (batchlane.Outcome, error)
	// Block, when non-nil, is waited on inside every call before it returns, so
	// a test can hold an item in flight while it cancels the batch.
	Block chan struct{}
}

// Fn returns the batchlane.DispatchFn to hand to batchlane.New.
func (f *FakeDispatch) Fn() batchlane.DispatchFn {
	return func(ctx context.Context, accountID, apiKeyID, model string, body []byte) (batchlane.Outcome, error) {
		f.mu.Lock()
		n := len(f.calls)
		f.calls = append(f.calls, DispatchCall{
			AccountID: accountID,
			APIKeyID:  apiKeyID,
			Model:     model,
			BodyLen:   len(body),
		})
		respond, block := f.Respond, f.Block
		f.mu.Unlock()

		if block != nil {
			select {
			case <-block:
			case <-ctx.Done():
				// The dispatch funnel reports a cancelled attempt rather than a
				// failure, so the dispatcher does not charge it an attempt.
				return batchlane.Outcome{ErrCode: batchlane.ErrCodeCancelled}, nil
			}
		}
		if ctx.Err() != nil {
			return batchlane.Outcome{ErrCode: batchlane.ErrCodeCancelled}, nil
		}
		if respond == nil {
			return batchlane.Outcome{ResponseBody: []byte(`{"ok":true}`)}, nil
		}
		return respond(n, model, body)
	}
}

// Calls returns a copy of everything the funnel was asked to run.
func (f *FakeDispatch) Calls() []DispatchCall {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]DispatchCall(nil), f.calls...)
}

// Len is the number of calls made so far.
func (f *FakeDispatch) Len() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

// FakeFinalize records the batches the dispatcher asked to finalize. PR2's
// assembler provides the real hook.
type FakeFinalize struct {
	mu    sync.Mutex
	calls []string
	// Err, when non-nil, is returned by every call.
	Err error
	// Then, when non-nil, runs after the call is recorded. The terminal status
	// transition belongs to finalize, so a test that asserts on a batch reaching
	// expired or cancelled has to supply the part of the assembler that performs
	// it; Then is where that goes.
	Then func(batchID string, now time.Time) error
}

// Fn returns the finalize hook to hand to batchlane.New.
func (f *FakeFinalize) Fn() func(batchID string, now time.Time) error {
	return func(batchID string, now time.Time) error {
		f.mu.Lock()
		f.calls = append(f.calls, batchID)
		err, then := f.Err, f.Then
		f.mu.Unlock()

		if err != nil {
			return err
		}
		if then != nil {
			return then(batchID, now)
		}
		return nil
	}
}

// Calls returns the batch ids finalize was called with, in order.
func (f *FakeFinalize) Calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.calls...)
}

// Called reports whether batchID was finalized at least once.
func (f *FakeFinalize) Called(batchID string) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	for _, id := range f.calls {
		if id == batchID {
			return true
		}
	}
	return false
}
