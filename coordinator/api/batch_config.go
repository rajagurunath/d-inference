package api

// Batch-lane storage configuration (docs/design/tidal-batch-lane.md §3.3).
//
// The batch lane is the one place prompts sit on coordinator disk, so it is off
// unless it has a key it can seal them with. Production derives that key from
// the same mnemonic the sender-encryption key comes from, under its own HKDF
// domain; local development may opt into a process-local random key, which is
// loudly warned about because every blob becomes unreadable on restart.

import (
	"errors"
	"log/slog"

	"github.com/eigeninference/d-inference/coordinator/env"
	"github.com/eigeninference/d-inference/coordinator/internal/e2e"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

// DefaultBatchBlobDir is the on-VM location of the sealed batch blobs. It sits
// under the same userdata mount the rest of the coordinator's durable state
// uses so a redeploy does not orphan in-flight batches.
const DefaultBatchBlobDir = "/mnt/disks/userdata/batch"

// BatchConfig holds the deployment knobs for the batch lane's blob storage.
// The zero value plus no mnemonic leaves the batch routes disabled.
type BatchConfig struct {
	// BlobDir is where sealed batch blobs are written (0700, files 0600).
	BlobDir string // EIGENINFERENCE_BATCH_BLOB_DIR
	// DevInsecureKey allows the coordinator to run the batch lane on a
	// process-local random key when no mnemonic is configured. Local
	// development only: blobs written under it are unreadable after a restart.
	DevInsecureKey bool // EIGENINFERENCE_BATCH_DEV_INSECURE_KEY
}

// ReadBatchConfig reads the batch-lane configuration from the environment.
func ReadBatchConfig() BatchConfig {
	return BatchConfig{
		BlobDir:        env.EnvOr(env.EnvPrefix+"_BATCH_BLOB_DIR", DefaultBatchBlobDir),
		DevInsecureKey: env.EnvBool(env.EnvPrefix+"_BATCH_DEV_INSECURE_KEY", false),
	}
}

// NewBatchBlobStore builds the sealed blob store the batch lane runs on.
//
// It returns (nil, nil) when no key is available — no mnemonic and no dev
// escape hatch — which is not an error: the coordinator still serves online
// traffic and every batch route answers 503 batch_unavailable. It returns an
// error only when a key exists but the directory cannot be prepared, because
// that is a misconfigured deployment rather than a disabled feature.
func NewBatchBlobStore(cfg BatchConfig, mnemonic string, logger *slog.Logger) (*sealedblob.Store, error) {
	key, err := sealedblob.DeriveKey(mnemonic)
	switch {
	case err == nil:
	case errors.Is(err, e2e.ErrNoMnemonic) && cfg.DevInsecureKey:
		if key, err = sealedblob.RandomKey(); err != nil {
			return nil, err
		}
		logger.Warn("batch lane running on a PROCESS-LOCAL RANDOM key — EIGENINFERENCE_BATCH_DEV_INSECURE_KEY is set; " +
			"every batch input and result becomes unreadable when this process exits. Never set this outside local development.")
	case errors.Is(err, e2e.ErrNoMnemonic):
		logger.Warn("batch lane disabled — no mnemonic configured, so batch inputs could not be sealed at rest " +
			"(set MNEMONIC, or EIGENINFERENCE_BATCH_DEV_INSECURE_KEY=true for local development)")
		return nil, nil
	default:
		return nil, err
	}

	dir := cfg.BlobDir
	if dir == "" {
		dir = DefaultBatchBlobDir
	}
	return sealedblob.New(dir, key)
}

// SetBatchBlobStore installs the sealed blob store backing the batch lane.
// Call once before serving; nil leaves the batch routes disabled.
func (s *Server) SetBatchBlobStore(bs *sealedblob.Store) {
	s.batchBlobs = bs
}

// BatchBlobs returns the sealed blob store backing the batch lane, or nil when
// the lane is not configured. The batch dispatcher reads item bodies and writes
// result blobs through it.
func (s *Server) BatchBlobs() *sealedblob.Store { return s.batchBlobs }

// BatchItemInputRef is the blob ref an item's sealed request body lives under.
// It is deliberately distinct from the result ref: finalize deletes every input
// blob as soon as a batch settles, and a shared ref would take the results with
// it. The suffix keeps the ref inside sealedblob's [A-Za-z0-9_-] shape.
func BatchItemInputRef(itemID string) string { return itemID + "-in" }

// BatchItemResultRef is the blob ref the batch dispatcher writes an item's
// sealed result under, before it settles the item. Rewriting it is idempotent,
// so a retried attempt is safe.
func BatchItemResultRef(itemID string) string { return itemID }

// batchStore returns the blob store, or a 503 batch_unavailable error when the
// lane has no key. Every batch handler starts here so a coordinator without a
// mnemonic never half-accepts a batch.
func (s *Server) batchStore() (*sealedblob.Store, error) {
	if s.batchBlobs == nil {
		return nil, &batchError{
			Status:  503,
			Type:    "batch_unavailable",
			Code:    "batch_unavailable",
			Message: "the batch lane is not configured on this coordinator",
		}
	}
	return s.batchBlobs, nil
}
