package batchlane

import (
	"io"
	"log/slog"
	"testing"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
)

// viewTestProvider registers a synthetic provider serving one model, the same
// shape registry's own scheduler fixtures build, driven entirely through the
// registry's exported surface so this test lives outside package registry.
func viewTestProvider(t *testing.T, reg *registry.Registry, id, model string, slot protocol.BackendSlotCapacity) *registry.Provider {
	t.Helper()
	msg := &protocol.RegisterMessage{
		Type:      protocol.TypeRegister,
		Models:    []protocol.ModelInfo{{ID: model, ModelType: "chat", Quantization: "4bit"}},
		DecodeTPS: 40,
		Hardware: protocol.Hardware{
			MachineModel: "Mac15,8",
			ChipName:     "Apple M3 Max",
			ChipFamily:   "M3",
			ChipTier:     "Max",
			MemoryGB:     64,
		},
	}
	p := reg.Register(id, nil, msg)
	slot.Model = model
	if slot.State == "" {
		slot.State = "running"
	}
	// The provider is not serving concurrently with this test, so its exported
	// capacity fields are set directly rather than through a heartbeat.
	p.BackendCapacity = &protocol.BackendCapacity{TotalMemoryGB: 64, Slots: []protocol.BackendSlotCapacity{slot}}
	return p
}

func viewTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	return registry.New(slog.New(slog.NewTextHandler(io.Discard, nil)))
}

// TestRegistryViewMapsSlotSignals is the adapter contract: every field the
// control law reads comes from the live heartbeat capacity report, and
// MaxPerSlot is the registry's OWN batch row allowance rather than a number
// batchlane derives for itself.
func TestRegistryViewMapsSlotSignals(t *testing.T) {
	reg := viewTestRegistry(t)
	reg.SetQualityConcurrencyCap(true, 1.2, 15, 4)
	model := "view-model"
	p := viewTestProvider(t, reg, "A", model, protocol.BackendSlotCapacity{
		NumRunning:            2,
		NumWaiting:            1,
		ObservedDecodeTPS:     22,
		ActiveTokenBudgetUsed: 300,
		ActiveTokenBudgetMax:  1000,
	})

	view := NewRegistryView(reg)
	slots := view.Slots(model)

	key := SlotKey{ProviderID: "A", Model: model}
	sig, ok := slots[key]
	if !ok {
		t.Fatalf("no signal for %+v; got %d slots", key, len(slots))
	}
	if sig.Running != 2 || sig.Waiting != 1 {
		t.Fatalf("running/waiting = %d/%d, want 2/1", sig.Running, sig.Waiting)
	}
	if sig.DecodeTPS != 22 {
		t.Fatalf("DecodeTPS = %v, want 22", sig.DecodeTPS)
	}
	if sig.KV != 0.3 || !sig.KVKnown {
		t.Fatalf("KV = %v (known %v), want 0.3 (300/1000) known", sig.KV, sig.KVKnown)
	}
	if sig.DecodeFloor != reg.QualityCapFloorTPS() {
		t.Fatalf("DecodeFloor = %v, want the registry floor %v", sig.DecodeFloor, reg.QualityCapFloorTPS())
	}
	if want := reg.BatchRowsAllowed(p, model); sig.MaxPerSlot != want {
		t.Fatalf("MaxPerSlot = %d, want BatchRowsAllowed %d", sig.MaxPerSlot, want)
	}
}

// A slot that reports no token budget has no KV signal to read. The view must
// say so rather than substitute a number: any substitute is wrong in one
// direction or the other (0 reads as idle and grows without bound, a hold-band
// value pins a fresh slot at target 0 for good).
func TestRegistryViewMarksAnAbsentKVBudgetUnknown(t *testing.T) {
	reg := viewTestRegistry(t)
	model := "view-nokv-model"
	viewTestProvider(t, reg, "A", model, protocol.BackendSlotCapacity{NumRunning: 1})

	sig := NewRegistryView(reg).Slots(model)[SlotKey{ProviderID: "A", Model: model}]
	if sig.KVKnown {
		t.Fatalf("KVKnown = true for a slot with no reported budget (KV %v)", sig.KV)
	}
}

func TestRegistryViewSkipsOtherModelsAndUnloadedSlots(t *testing.T) {
	reg := viewTestRegistry(t)
	model := "view-wanted-model"
	viewTestProvider(t, reg, "A", model, protocol.BackendSlotCapacity{})
	viewTestProvider(t, reg, "B", "view-other-model", protocol.BackendSlotCapacity{})
	viewTestProvider(t, reg, "C", model, protocol.BackendSlotCapacity{State: "crashed"})

	slots := NewRegistryView(reg).Slots(model)
	if len(slots) != 1 {
		t.Fatalf("got %d slots, want 1: %+v", len(slots), slots)
	}
	if _, ok := slots[SlotKey{ProviderID: "A", Model: model}]; !ok {
		t.Fatalf("expected only provider A's slot, got %+v", slots)
	}
}

// The empty model means "every slot in the fleet" — what the dispatcher asks
// for, because placement is the reservation path's decision and the tick only
// needs a fleet-wide in-flight budget.
func TestRegistryViewEmptyModelReturnsEveryLoadedSlot(t *testing.T) {
	reg := viewTestRegistry(t)
	viewTestProvider(t, reg, "A", "view-a-model", protocol.BackendSlotCapacity{})
	viewTestProvider(t, reg, "B", "view-b-model", protocol.BackendSlotCapacity{})

	if slots := NewRegistryView(reg).Slots(""); len(slots) != 2 {
		t.Fatalf("got %d slots, want 2: %+v", len(slots), slots)
	}
}

func TestRegistryViewOnNilRegistryIsEmpty(t *testing.T) {
	if slots := NewRegistryView(nil).Slots(""); len(slots) != 0 {
		t.Fatalf("got %d slots, want none", len(slots))
	}
}
