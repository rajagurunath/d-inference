package batchlane

// registry_view.go adapts the provider registry to the control law's input.
// It is the only place batchlane reads the registry, and it reads it through
// registry's own snapshot accessor rather than by walking live providers, so no
// registry mutex is ever held across a dispatcher tick.

import "github.com/eigeninference/d-inference/coordinator/registry"

// SlotKey identifies one provider·model slot across ticks, so the dispatcher's
// per-slot AIMD state survives the snapshot being rebuilt every second.
type SlotKey struct {
	ProviderID string
	Model      string
}

// RegistryView is the dispatcher's window onto live fleet capacity. The
// dispatcher calls Slots("") for a fleet-wide view; tests substitute a fake.
type RegistryView interface {
	// Slots returns one signal per loaded slot serving model, or per loaded
	// slot in the fleet when model is empty.
	Slots(model string) map[SlotKey]SlotSignal
}

// unknownKV is the KV pressure reported for a slot whose provider publishes no
// token budget. It sits inside the controller's hold band (KVLow, KVHigh], so
// a slot with no KV signal neither grows nor shrinks the target on that
// account — reporting 0 would read as "idle" and drive the target up forever.
const unknownKV = (KVLow + KVHigh) / 2

// NewRegistryView returns the live view over reg. A nil registry yields an
// empty view rather than a panic, so a coordinator that starts the lane before
// the registry exists simply dispatches nothing.
func NewRegistryView(reg *registry.Registry) RegistryView { return registryView{reg: reg} }

type registryView struct{ reg *registry.Registry }

func (v registryView) Slots(model string) map[SlotKey]SlotSignal {
	if v.reg == nil {
		return map[SlotKey]SlotSignal{}
	}
	floor := v.reg.QualityCapFloorTPS()
	slots := v.reg.BatchSlots(model)
	out := make(map[SlotKey]SlotSignal, len(slots))
	for _, s := range slots {
		out[SlotKey{ProviderID: s.ProviderID, Model: s.Model}] = SlotSignal{
			Waiting:     s.NumWaiting,
			DecodeTPS:   s.ObservedDecodeTPS,
			DecodeFloor: floor,
			KV:          kvPressure(s.ActiveTokenBudgetUsed, s.ActiveTokenBudgetMax),
			Running:     s.NumRunning,
			MaxPerSlot:  s.BatchRowsAllowed,
		}
	}
	return out
}

// kvPressure is used/max clamped to [0, 1], or unknownKV when the provider
// reports no budget.
func kvPressure(used, max int64) float64 {
	if max <= 0 {
		return unknownKV
	}
	kv := float64(used) / float64(max)
	if kv < 0 {
		return 0
	}
	if kv > 1 {
		return 1
	}
	return kv
}
