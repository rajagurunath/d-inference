package sealedblob

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/eigeninference/d-inference/coordinator/internal/e2e"
)

const testMnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"

func TestPutPlainThenOpenRoundTrips(t *testing.T) {
	k, err := RandomKey()
	if err != nil {
		t.Fatalf("RandomKey: %v", err)
	}
	s, err := New(t.TempDir(), k)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := s.PutPlain("a", []byte(`{"x":1}`)); err != nil {
		t.Fatalf("PutPlain: %v", err)
	}
	got, err := s.Open("a")
	if err != nil || string(got) != `{"x":1}` {
		t.Fatalf("Open = %q, err %v; want %q", got, err, `{"x":1}`)
	}
	info, err := os.Stat(filepath.Join(s.Dir(), "a"))
	if err != nil {
		t.Fatalf("stat blob: %v", err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("blob mode = %v, want 0600", info.Mode().Perm())
	}
	dirInfo, err := os.Stat(s.Dir())
	if err != nil {
		t.Fatalf("stat dir: %v", err)
	}
	if dirInfo.Mode().Perm() != 0o700 {
		t.Fatalf("dir mode = %v, want 0700", dirInfo.Mode().Perm())
	}
}

func TestPutToRecipientIsUnreadableByStore(t *testing.T) {
	k, _ := RandomKey()
	s, err := New(t.TempDir(), k)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	other, _ := RandomKey()
	if err := s.PutTo("r", []byte("secret"), other.Public); err != nil {
		t.Fatalf("PutTo: %v", err)
	}
	if _, err := s.Open("r"); err == nil {
		t.Fatal("store must not open a blob sealed to the consumer")
	}
	raw, err := s.Raw("r")
	if err != nil {
		t.Fatalf("Raw: %v", err)
	}
	var payload e2e.EncryptedPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("on-disk format is not an e2e.EncryptedPayload: %v", err)
	}
	plain, err := e2e.DecryptWithPrivateKey(&payload, other.Private)
	if err != nil {
		t.Fatalf("recipient could not open its own blob: %v", err)
	}
	if string(plain) != "secret" {
		t.Fatalf("recipient plaintext = %q, want %q", plain, "secret")
	}
}

func TestDeriveKeyIsDeterministicAndDomainSeparated(t *testing.T) {
	a, err := DeriveKey(testMnemonic)
	if err != nil {
		t.Fatalf("DeriveKey: %v", err)
	}
	b, err := DeriveKey(testMnemonic)
	if err != nil {
		t.Fatalf("DeriveKey (second): %v", err)
	}
	if a.Public != b.Public || a.Private != b.Private {
		t.Fatal("DeriveKey is not deterministic")
	}
	ck, err := e2e.DeriveCoordinatorKey(testMnemonic)
	if err != nil {
		t.Fatalf("DeriveCoordinatorKey: %v", err)
	}
	if ck.PublicKey == a.Public {
		t.Fatal("batch-store key must be domain-separated from the e2e coordinator key")
	}
}

func TestDeriveKeyRejectsEmptyMnemonic(t *testing.T) {
	if _, err := DeriveKey("  "); err == nil {
		t.Fatal("DeriveKey must reject an empty mnemonic")
	}
}

func TestRawIsCiphertext(t *testing.T) {
	k, _ := RandomKey()
	s, err := New(t.TempDir(), k)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	plaintext := []byte("the quick brown fox")
	if err := s.PutPlain("c", plaintext); err != nil {
		t.Fatalf("PutPlain: %v", err)
	}
	raw, err := s.Raw("c")
	if err != nil {
		t.Fatalf("Raw: %v", err)
	}
	if bytes.Contains(raw, plaintext) {
		t.Fatal("Raw returned plaintext bytes")
	}
	onDisk, err := os.ReadFile(filepath.Join(s.Dir(), "c"))
	if err != nil {
		t.Fatalf("read blob: %v", err)
	}
	if bytes.Contains(onDisk, plaintext) {
		t.Fatal("blob on disk contains plaintext bytes")
	}
	if !bytes.Equal(raw, onDisk) {
		t.Fatal("Raw must return the sealed bytes verbatim")
	}
}

func TestDeleteRemovesBlobAndIsIdempotent(t *testing.T) {
	k, _ := RandomKey()
	s, err := New(t.TempDir(), k)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := s.PutPlain("d", []byte("x")); err != nil {
		t.Fatalf("PutPlain: %v", err)
	}
	if err := s.Delete("d"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if err := s.Delete("d"); err != nil {
		t.Fatalf("Delete must be idempotent, got %v", err)
	}
	if _, err := s.Open("d"); err == nil {
		t.Fatal("Open must fail after Delete")
	}
}

func TestRefsAreRejectedWhenNotIDShaped(t *testing.T) {
	k, _ := RandomKey()
	s, err := New(t.TempDir(), k)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, ref := range []string{"", ".", "..", "a/b", "../escape", "a b", "a.json"} {
		if err := s.PutPlain(ref, []byte("x")); err == nil {
			t.Fatalf("PutPlain(%q) must be rejected", ref)
		}
		if _, err := s.Open(ref); err == nil {
			t.Fatalf("Open(%q) must be rejected", ref)
		}
	}
}

func TestPutPlainOverwritesInPlace(t *testing.T) {
	k, _ := RandomKey()
	s, err := New(t.TempDir(), k)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := s.PutPlain("o", []byte("first")); err != nil {
		t.Fatalf("PutPlain: %v", err)
	}
	if err := s.PutPlain("o", []byte("second")); err != nil {
		t.Fatalf("PutPlain (overwrite): %v", err)
	}
	got, err := s.Open("o")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if string(got) != "second" {
		t.Fatalf("Open = %q, want %q", got, "second")
	}
	entries, err := os.ReadDir(s.Dir())
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("dir has %d entries, want 1 (no temp files left behind)", len(entries))
	}
}

// TestPutPlainSyncsBlobAndDirectory exercises the fsync-then-rename-then-fsync-
// directory path in write: the temp file and the directory entry it publishes
// into must both survive the call and nothing but the published blob is left
// behind.
func TestPutPlainSyncsBlobAndDirectory(t *testing.T) {
	k, _ := RandomKey()
	dir := t.TempDir()
	s, err := New(dir, k)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := s.PutPlain("a", []byte("payload")); err != nil {
		t.Fatalf("PutPlain: %v", err)
	}
	got, err := s.Open("a")
	if err != nil || string(got) != "payload" {
		t.Fatalf("Open = %q, err %v; want %q", got, err, "payload")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	if len(entries) != 1 || entries[0].Name() != "a" {
		t.Fatalf("dir entries = %v, want only the published blob", entries)
	}
}
