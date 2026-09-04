package api

import (
	"bytes"
	"log/slog"
	"strings"
	"sync"
	"testing"
)

// testMnemonic is the all-zero BIP39 vector; it derives a real batch-store key
// without any production secret.
const testMnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"

// syncBuffer collects log output for assertions under -race.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func TestNewBatchBlobStoreDerivesFromMnemonic(t *testing.T) {
	dir := t.TempDir()
	var logs syncBuffer
	logger := slog.New(slog.NewTextHandler(&logs, nil))

	bs, err := NewBatchBlobStore(BatchConfig{BlobDir: dir}, testMnemonic, logger)
	if err != nil {
		t.Fatalf("NewBatchBlobStore: %v", err)
	}
	if bs == nil {
		t.Fatal("a configured mnemonic must produce a store")
	}
	if bs.Dir() != dir {
		t.Fatalf("dir = %q, want %q", bs.Dir(), dir)
	}
	if strings.Contains(logs.String(), "RANDOM key") {
		t.Fatalf("a mnemonic-derived key must not warn about the dev key: %s", logs.String())
	}
}

func TestNewBatchBlobStoreDisabledWithoutMnemonic(t *testing.T) {
	var logs syncBuffer
	logger := slog.New(slog.NewTextHandler(&logs, nil))

	bs, err := NewBatchBlobStore(BatchConfig{BlobDir: t.TempDir()}, "", logger)
	if err != nil {
		t.Fatalf("a missing mnemonic is not an error: %v", err)
	}
	if bs != nil {
		t.Fatal("no mnemonic and no dev key must leave the lane disabled")
	}
	if !strings.Contains(logs.String(), "batch lane disabled") {
		t.Fatalf("want a disabled warning, got %s", logs.String())
	}
}

func TestNewBatchBlobStoreDevInsecureKeyWarnsOnce(t *testing.T) {
	var logs syncBuffer
	logger := slog.New(slog.NewTextHandler(&logs, nil))

	bs, err := NewBatchBlobStore(BatchConfig{BlobDir: t.TempDir(), DevInsecureKey: true}, "", logger)
	if err != nil {
		t.Fatalf("NewBatchBlobStore: %v", err)
	}
	if bs == nil {
		t.Fatal("the dev escape hatch must produce a store")
	}
	if n := strings.Count(logs.String(), "PROCESS-LOCAL RANDOM key"); n != 1 {
		t.Fatalf("want exactly one dev-key WARN, got %d in %s", n, logs.String())
	}
	if !strings.Contains(logs.String(), "level=WARN") {
		t.Fatalf("the dev-key line must be a WARN, got %s", logs.String())
	}
}

func TestBatchRoutesReportUnavailableWithoutAKey(t *testing.T) {
	srv, _ := testServer(t)
	if _, err := srv.batchStore(); err == nil {
		t.Fatal("a server with no batch key must refuse batch work")
	} else {
		be := batchErrorFrom(t, err)
		if be.Status != 503 || be.Code != "batch_unavailable" {
			t.Fatalf("got %d/%s, want 503/batch_unavailable", be.Status, be.Code)
		}
	}
}
