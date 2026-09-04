// Package sealedblob stores batch inputs and outputs on coordinator disk as
// NaCl-Box ciphertext.
//
// The threat model assumes prompts never touch coordinator disk in the clear;
// the batch lane has to hold items for up to 24 hours, so every blob is sealed
// with a fresh ephemeral key to a long-lived X25519 batch-store key derived
// from the coordinator mnemonic with its own HKDF domain (HKDFInfo below), the
// same construction internal/e2e uses for the sender-sealing key. Results may
// instead be sealed to a consumer-supplied key, in which case the coordinator
// can no longer read them.
//
// The on-disk format is the JSON encoding of e2e.EncryptedPayload. Blob refs
// are item or file ids only — never a custom_id, a filename, or anything else
// derived from user input — so a directory listing carries no content.
package sealedblob

import (
	"crypto/sha256"
	"crypto/sha512"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/hkdf"
	"golang.org/x/crypto/pbkdf2"

	"github.com/eigeninference/d-inference/coordinator/internal/e2e"
)

// HKDFInfo is the HKDF info string that separates the batch-store key from the
// sender-sealing key (e2e.CoordinatorKeyHKDFInfo) derived from the same
// mnemonic. Bumping the version here rotates the key and orphans every blob
// still on disk.
const HKDFInfo = "eigeninference-coordinator-batchstore-v1"

// ErrNotFound is wrapped by Open, Raw, and Delete when no blob exists for the
// ref. Callers that distinguish a purged blob from a disk failure must check
// with errors.Is.
var ErrNotFound = errors.New("sealedblob: not found")

// Key is the long-lived X25519 keypair blobs are sealed to.
type Key struct {
	Public  [32]byte
	Private [32]byte
}

// DeriveKey derives the batch-store keypair from a BIP39 mnemonic. It mirrors
// e2e.DeriveCoordinatorKey with HKDFInfo swapped in, so the two keys are
// unrelated even though they share a seed.
func DeriveKey(mnemonic string) (*Key, error) {
	mnemonic = strings.TrimSpace(mnemonic)
	if mnemonic == "" {
		return nil, e2e.ErrNoMnemonic
	}

	words := strings.Fields(mnemonic)
	if len(words) != 12 && len(words) != 24 {
		return nil, fmt.Errorf("sealedblob: mnemonic must be 12 or 24 words, got %d", len(words))
	}

	seed := pbkdf2.Key([]byte(mnemonic), []byte("mnemonic"), 2048, 64, sha512.New)

	r := hkdf.New(sha256.New, seed, nil, []byte(HKDFInfo))
	var priv [32]byte
	if _, err := io.ReadFull(r, priv[:]); err != nil {
		return nil, fmt.Errorf("sealedblob: hkdf read: %w", err)
	}

	pub, err := curve25519.X25519(priv[:], curve25519.Basepoint)
	if err != nil {
		return nil, fmt.Errorf("sealedblob: derive x25519 public key: %w", err)
	}

	k := &Key{Private: priv}
	copy(k.Public[:], pub)
	return k, nil
}

// RandomKey generates a process-local keypair. Blobs sealed to it are
// unreadable after a restart, so it is for local development only and the
// caller is expected to log that loudly.
func RandomKey() (*Key, error) {
	session, err := e2e.GenerateSessionKeys()
	if err != nil {
		return nil, fmt.Errorf("sealedblob: generate random key: %w", err)
	}
	return &Key{Public: session.PublicKey, Private: session.PrivateKey}, nil
}

// Store is a directory of sealed blobs, one file per ref.
type Store struct {
	dir string
	key *Key
}

// New creates the blob directory (mode 0700) if it does not exist and returns
// a store sealing to key.
func New(dir string, key *Key) (*Store, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, errors.New("sealedblob: directory is required")
	}
	if key == nil {
		return nil, errors.New("sealedblob: key is required")
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("sealedblob: create blob directory: %w", err)
	}
	// MkdirAll leaves an existing directory's mode alone; tighten it so an
	// upgrade from a looser deployment does not leave blobs world-readable.
	if err := os.Chmod(dir, 0o700); err != nil {
		return nil, fmt.Errorf("sealedblob: tighten blob directory mode: %w", err)
	}
	return &Store{dir: dir, key: key}, nil
}

// Dir returns the directory blobs are written to.
func (s *Store) Dir() string { return s.dir }

// PublicKey returns the batch-store public key blobs are sealed to by PutPlain.
func (s *Store) PublicKey() [32]byte { return s.key.Public }

// PutPlain seals plaintext to the store's own key and writes it under ref.
func (s *Store) PutPlain(ref string, plaintext []byte) error {
	return s.PutTo(ref, plaintext, s.key.Public)
}

// PutTo seals plaintext to recipient and writes it under ref. When recipient is
// a consumer-supplied key the store can no longer Open the blob; only Raw and
// Delete remain useful.
func (s *Store) PutTo(ref string, plaintext []byte, recipient [32]byte) error {
	path, err := s.path(ref)
	if err != nil {
		return err
	}
	session, err := e2e.GenerateSessionKeys()
	if err != nil {
		return fmt.Errorf("sealedblob: generate ephemeral key for %q: %w", ref, err)
	}
	payload, err := e2e.Encrypt(plaintext, recipient, session)
	if err != nil {
		return fmt.Errorf("sealedblob: seal %q: %w", ref, err)
	}
	sealed, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("sealedblob: encode %q: %w", ref, err)
	}
	return s.write(path, sealed)
}

// Open reads the blob at ref and decrypts it with the store's private key. It
// fails when the blob was sealed to another key (PutTo with a consumer key).
func (s *Store) Open(ref string) ([]byte, error) {
	sealed, err := s.Raw(ref)
	if err != nil {
		return nil, err
	}
	var payload e2e.EncryptedPayload
	if err := json.Unmarshal(sealed, &payload); err != nil {
		return nil, fmt.Errorf("sealedblob: decode %q: %w", ref, err)
	}
	plaintext, err := e2e.DecryptWithPrivateKey(&payload, s.key.Private)
	if err != nil {
		return nil, fmt.Errorf("sealedblob: open %q: %w", ref, err)
	}
	return plaintext, nil
}

// Raw returns the sealed bytes verbatim, for a /content download that hands the
// ciphertext to a consumer holding the recipient key.
func (s *Store) Raw(ref string) ([]byte, error) {
	path, err := s.path(ref)
	if err != nil {
		return nil, err
	}
	sealed, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("sealedblob: %q: %w", ref, ErrNotFound)
		}
		return nil, fmt.Errorf("sealedblob: read %q: %w", ref, err)
	}
	return sealed, nil
}

// Delete removes the blob at ref. Deleting a ref that is already gone is not an
// error, so retention sweeps and finalization can both run without coordination.
func (s *Store) Delete(ref string) error {
	path, err := s.path(ref)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("sealedblob: delete %q: %w", ref, err)
	}
	return nil
}

// write publishes sealed at path via a temp file + rename so a crash mid-write
// never leaves a truncated blob that Open would report as tampering.
func (s *Store) write(path string, sealed []byte) error {
	tmp, err := os.CreateTemp(s.dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("sealedblob: create temp blob: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return fmt.Errorf("sealedblob: set blob mode: %w", err)
	}
	if _, err := tmp.Write(sealed); err != nil {
		tmp.Close()
		return fmt.Errorf("sealedblob: write blob: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("sealedblob: close blob: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("sealedblob: publish blob: %w", err)
	}
	return nil
}

// path validates ref and joins it to the blob directory. Refs are ids minted by
// the store (file-…, bitem-…), so anything outside that shape — a path
// separator, a filename, a custom_id — is a bug and is rejected rather than
// echoed into an error message.
func (s *Store) path(ref string) (string, error) {
	if !validRef(ref) {
		return "", fmt.Errorf("sealedblob: invalid blob ref (want 1-128 chars of [A-Za-z0-9_-])")
	}
	return filepath.Join(s.dir, ref), nil
}

func validRef(ref string) bool {
	if ref == "" || len(ref) > 128 {
		return false
	}
	for _, c := range ref {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_', c == '-':
		default:
			return false
		}
	}
	return true
}
