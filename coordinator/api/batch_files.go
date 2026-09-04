package api

// /v1/files — batch input upload and retrieval (docs/design/tidal-batch-lane.md §3.6).
//
// An upload is validated in full before a byte of it is stored, then written as
// one sealed blob keyed by the file id, so a file that never becomes a batch is
// still ciphertext at rest. Two request shapes are accepted:
//
//   - multipart/form-data with purpose=batch and a JSONL file part (what every
//     OpenAI SDK sends), plaintext over TLS;
//   - a sealed JSON envelope carrying {purpose, filename, content_base64},
//     unsealed in memory by sealedTransport, so the plaintext never reaches
//     disk. sealedTransport's own fixed 16 MiB cap on the outer sealed body,
//     once the ciphertext and content_base64 base64 layers both eat into it,
//     leaves room for only about 9 MiB of actual file content; this path
//     enforces an explicit 8 MiB file cap of its own (maxSealedEnvelopeFileBytes)
//     so an oversized sealed upload gets one clear 413 instead of failing
//     confusingly upstream.
//
// The multipart path is read with mime/multipart's low-level Reader, not
// r.ParseMultipartForm: ParseMultipartForm spills any part over its memory
// threshold to an unencrypted os.CreateTemp file, which would put a batch
// prompt on disk in the clear before it is ever sealed. Streaming each part by
// hand keeps the plaintext in this process's memory only.
//
// Privacy: filenames, custom_ids, and body bytes never reach a log line. Only
// ids, byte counts, and line counts do.

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

// Batch file purposes. Only "batch" may be uploaded; the other two are minted
// by the assembler when a batch finalizes.
const (
	batchFilePurposeInput  = "batch"
	batchFilePurposeOutput = "batch_output"
	batchFilePurposeError  = "batch_error"
)

// maxBatchFilenameLen bounds the stored filename. It is consumer text that is
// echoed back on GET /v1/files/{id} and nowhere else.
const maxBatchFilenameLen = 255

// multipartSlackBytes covers the MIME part headers and boundaries wrapping a
// maxFileBytes payload; base64SlackNumer/Denom cover the 4/3 inflation of a
// base64 envelope. Both keep a legal maximum-size upload from tripping the
// transport cap before the handler can report a precise error.
const multipartSlackBytes = 1 << 20

// newBatchID mints an id with the given prefix and 24 hex characters of
// entropy. The shape matches sealedblob's accepted blob refs, so an id is
// always a legal blob name and consumer input never is.
func newBatchID(prefix string) (string, error) {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("mint %s id: %w", prefix, err)
	}
	return prefix + hex.EncodeToString(b[:]), nil
}

// batchModelResolver adapts the coordinator's alias resolution for the batch
// parser: a public alias becomes the concrete build id the dispatcher will use
// hours later, and anything the catalog does not know is refused now rather
// than after the consumer has gone away.
func (s *Server) batchModelResolver() modelResolver {
	return func(requested string) (string, bool) {
		requested = strings.TrimSpace(requested)
		if requested == "" {
			return "", false
		}
		buildModel, _, _, ok := s.resolveRequestedModel(
			map[string]any{}, nil, requested, nil, selfRoutePolicy{}, registry.RequestTraits{})
		if !ok || buildModel == "" {
			return "", false
		}
		if !s.registry.IsModelInCatalog(buildModel) {
			return "", false
		}
		return buildModel, true
	}
}

// batchFileObject renders the OpenAI file object.
func batchFileObject(f *store.BatchFile) map[string]any {
	return map[string]any{
		"object":     "file",
		"id":         f.ID,
		"purpose":    f.Purpose,
		"filename":   f.Filename,
		"bytes":      f.SizeBytes,
		"created_at": f.CreatedAt.Unix(),
		"status":     "processed",
	}
}

// handleBatchFileUpload handles POST /v1/files.
func (s *Server) handleBatchFileUpload(w http.ResponseWriter, r *http.Request) {
	blobs, err := s.batchStore()
	if err != nil {
		s.writeBatchError(w, err)
		return
	}
	accountID := s.resolveAccountID(r)

	purpose, filename, content, err := s.readBatchUpload(w, r)
	if err != nil {
		s.writeBatchError(w, err)
		return
	}
	if purpose != batchFilePurposeInput {
		s.writeBatchError(w, batchErr("invalid_purpose", "purpose",
			"purpose must be %q", batchFilePurposeInput))
		return
	}
	if int64(len(content)) > maxFileBytes {
		s.writeBatchError(w, &batchError{
			Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error",
			Code: "file_too_large", Param: "file",
			Message: fmt.Sprintf("input file exceeds the %d-byte batch upload limit", maxFileBytes),
		})
		return
	}

	// Validate every line before anything is stored: a file that cannot become
	// a batch must not occupy disk, and a half-valid file must not be
	// discoverable through /v1/files.
	items, err := parseBatchJSONL(strings.NewReader(string(content)), "", maxFileLines, s.batchModelResolver())
	if err != nil {
		s.writeBatchError(w, err)
		return
	}

	fileID, err := newBatchID("file-")
	if err != nil {
		s.writeBatchError(w, s.internalBatchError(err))
		return
	}
	if err := blobs.PutPlain(fileID, content); err != nil {
		s.writeBatchError(w, s.internalBatchError(err, "file_id", fileID))
		return
	}
	rec := &store.BatchFile{
		ID:        fileID,
		AccountID: accountID,
		Purpose:   batchFilePurposeInput,
		Filename:  filename,
		SizeBytes: int64(len(content)),
		CreatedAt: time.Now().UTC(),
		BlobRef:   fileID,
		SealedBy:  "coordinator",
	}
	if err := s.store.CreateBatchFile(rec); err != nil {
		// The blob is orphaned without a row pointing at it; drop it now rather
		// than leaving ciphertext no retention sweep can reach.
		if delErr := blobs.Delete(fileID); delErr != nil {
			s.logger.Error("batch: removing an orphaned input blob failed", "file_id", fileID, "error", delErr)
		}
		s.writeBatchError(w, s.internalBatchError(err, "file_id", fileID))
		return
	}

	s.logger.Info("batch: input file accepted",
		"file_id", fileID, "account_id", accountID, "bytes", len(content), "requests", len(items))
	writeJSON(w, http.StatusOK, batchFileObject(rec))
}

// readBatchUpload pulls the purpose, filename, and plaintext JSONL out of
// either accepted request shape.
func (s *Server) readBatchUpload(w http.ResponseWriter, r *http.Request) (purpose, filename string, content []byte, err error) {
	mediaType, _, parseErr := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if parseErr != nil {
		mediaType = ""
	}
	if strings.HasPrefix(mediaType, "multipart/") {
		return s.readMultipartUpload(w, r)
	}
	return s.readEnvelopeUpload(w, r)
}

// maxMultipartFieldBytes bounds a non-file form field (only "purpose" is
// expected). It has nothing to do with the file cap; it just keeps a
// misbehaving client from streaming an unbounded field value into memory
// under the guise of a form field.
const maxMultipartFieldBytes = 1 << 10

// readMultipartUpload streams the request through mime/multipart's low-level
// Reader instead of r.ParseMultipartForm: ParseMultipartForm buffers each part
// past its memory threshold to a plaintext temp file via os.CreateTemp, which
// would put an uploaded prompt on disk in the clear before this handler ever
// seals it. Reading each part directly means the file's bytes exist only in
// this function's own memory, in an io.LimitReader-bounded read, and never
// touch $TMPDIR.
func (s *Server) readMultipartUpload(w http.ResponseWriter, r *http.Request) (string, string, []byte, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxFileBytes+multipartSlackBytes)
	mr, err := r.MultipartReader()
	if err != nil {
		return "", "", nil, batchErr("invalid_request", "file", "multipart form could not be read")
	}

	var (
		purpose  string
		filename string
		content  []byte
		haveFile bool
	)
	for {
		part, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			if tooLarge(err) {
				return "", "", nil, oversizeBatchUpload()
			}
			return "", "", nil, batchErr("invalid_request", "file", "multipart form could not be read")
		}

		switch part.FormName() {
		case "purpose":
			raw, err := io.ReadAll(io.LimitReader(part, maxMultipartFieldBytes))
			part.Close()
			if err != nil {
				if tooLarge(err) {
					return "", "", nil, oversizeBatchUpload()
				}
				return "", "", nil, batchErr("invalid_request", "purpose", "the purpose field could not be read")
			}
			purpose = strings.TrimSpace(string(raw))
		case "file":
			if haveFile {
				// A second file part is ignored; the first one wins.
				part.Close()
				continue
			}
			body, err := io.ReadAll(io.LimitReader(part, maxFileBytes+1))
			partFilename := part.FileName()
			part.Close()
			if err != nil {
				if tooLarge(err) {
					return "", "", nil, oversizeBatchUpload()
				}
				return "", "", nil, batchErr("invalid_request", "file", "the uploaded file could not be read")
			}
			filename = sanitizeBatchFilename(partFilename)
			content = body
			haveFile = true
		default:
			part.Close()
		}
	}
	if !haveFile {
		return "", "", nil, batchErr("invalid_request", "file", "a JSONL file part named \"file\" is required")
	}
	return purpose, filename, content, nil
}

// sealedFileEnvelope is the JSON body shape POST /v1/files accepts, either as
// plaintext JSON or unsealed in memory by sealedTransport.
type sealedFileEnvelope struct {
	Purpose       string `json:"purpose"`
	Filename      string `json:"filename"`
	ContentBase64 string `json:"content_base64"`
}

// maxSealedEnvelopeFileBytes is the file-content cap on this path when it is
// reached through sealedTransport. sealedTransport caps the whole outer sealed
// request at a fixed 16 MiB (sender_encryption.go); that budget also has to
// pay for the outer envelope's own JSON, the NaCl-Box nonce and MAC, one layer
// of base64 for the ciphertext, and — inside the plaintext it decrypts to —
// the content_base64 field's own base64 inflation. Working that through
// leaves roughly 9 MiB of actual file content, so 8 MiB is enforced here
// explicitly, with its own precise 413, rather than letting an oversized
// sealed upload fail confusingly upstream (a generic body-too-large 400 from
// sealedTransport, or a truncated envelope that never parses as JSON).
const maxSealedEnvelopeFileBytes = 8 << 20

func oversizeSealedBatchUpload() *batchError {
	return &batchError{
		Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error",
		Code: "file_too_large", Param: "content_base64",
		Message: fmt.Sprintf("a batch file sent through the sealed envelope path is limited to %d bytes", maxSealedEnvelopeFileBytes),
	}
}

func (s *Server) readEnvelopeUpload(w http.ResponseWriter, r *http.Request) (string, string, []byte, error) {
	// Base64 inflates by 4/3; allow that plus the JSON wrapper so a legal
	// maximum-size file reaches the handler's own size check.
	r.Body = http.MaxBytesReader(w, r.Body, maxFileBytes*4/3+multipartSlackBytes)
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		if tooLarge(err) {
			return "", "", nil, oversizeBatchUpload()
		}
		return "", "", nil, batchErr("invalid_request", "", "request body could not be read")
	}
	var env sealedFileEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return "", "", nil, batchErr("invalid_request", "",
			"body must be multipart/form-data or a JSON object with purpose, filename, and content_base64")
	}
	// Reject an oversized sealed upload on the encoded length, before paying
	// for a base64 decode of something that can never fit.
	if isSealedRequest(r) && len(env.ContentBase64) > base64.StdEncoding.EncodedLen(maxSealedEnvelopeFileBytes) {
		return "", "", nil, oversizeSealedBatchUpload()
	}
	content, err := base64.StdEncoding.DecodeString(env.ContentBase64)
	if err != nil {
		return "", "", nil, batchErr("invalid_request", "content_base64", "content_base64 is not valid base64")
	}
	if isSealedRequest(r) && int64(len(content)) > maxSealedEnvelopeFileBytes {
		return "", "", nil, oversizeSealedBatchUpload()
	}
	return strings.TrimSpace(env.Purpose), sanitizeBatchFilename(env.Filename), content, nil
}

// sanitizeBatchFilename bounds the stored filename and strips control
// characters, so an echoed filename can never inject into a consumer's terminal
// or a downstream log.
func sanitizeBatchFilename(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "batch.jsonl"
	}
	name = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, name)
	if len(name) > maxBatchFilenameLen {
		name = name[:maxBatchFilenameLen]
	}
	if name == "" {
		return "batch.jsonl"
	}
	return name
}

func tooLarge(err error) bool {
	var maxErr *http.MaxBytesError
	return errors.As(err, &maxErr) || strings.Contains(err.Error(), "http: request body too large")
}

func oversizeBatchUpload() *batchError {
	return &batchError{
		Status: http.StatusRequestEntityTooLarge, Type: "invalid_request_error",
		Code: "file_too_large", Param: "file",
		Message: fmt.Sprintf("input file exceeds the %d-byte batch upload limit", maxFileBytes),
	}
}

// internalBatchError logs cause at error level — together with only the ids
// the caller passes in kv, never a consumer string — and returns a batchError
// so writeBatchError renders a 500 without leaking the cause to the consumer.
func (s *Server) internalBatchError(cause error, kv ...any) *batchError {
	args := append([]any{"error", cause}, kv...)
	s.logger.Error("batch: internal error", args...)
	return &batchError{
		Status: http.StatusInternalServerError, Type: "internal_error", Code: "internal_error",
		Message: "internal server error",
	}
}

// handleBatchFileGet handles GET /v1/files/{id}.
func (s *Server) handleBatchFileGet(w http.ResponseWriter, r *http.Request) {
	if _, err := s.batchStore(); err != nil {
		s.writeBatchError(w, err)
		return
	}
	f, ok := s.store.GetBatchFile(s.resolveAccountID(r), strings.TrimSpace(r.PathValue("id")))
	if !ok {
		s.writeBatchError(w, batchNotFound("file"))
		return
	}
	writeJSON(w, http.StatusOK, batchFileObject(f))
}

// handleBatchFileContent handles GET /v1/files/{id}/content. It returns the
// file's plaintext bytes over TLS; the blob itself stays sealed on disk. For a
// batch whose results are sealed to a consumer key, each output line's
// response.body is an e2e.EncryptedPayload only that consumer can open.
func (s *Server) handleBatchFileContent(w http.ResponseWriter, r *http.Request) {
	blobs, err := s.batchStore()
	if err != nil {
		s.writeBatchError(w, err)
		return
	}
	accountID := s.resolveAccountID(r)
	f, ok := s.store.GetBatchFile(accountID, strings.TrimSpace(r.PathValue("id")))
	if !ok {
		s.writeBatchError(w, batchNotFound("file"))
		return
	}
	if f.PurgedAt != nil {
		s.writeBatchError(w, &batchError{
			Status: http.StatusNotFound, Type: "invalid_request_error", Code: "file_content_purged",
			Message: "this file's content has been purged; only its metadata remains",
		})
		return
	}
	content, err := blobs.Open(f.BlobRef)
	if err != nil {
		if errors.Is(err, sealedblob.ErrNotFound) {
			s.writeBatchError(w, &batchError{
				Status: http.StatusNotFound, Type: "invalid_request_error", Code: "file_content_purged",
				Message: "this file's content has been purged; only its metadata remains",
			})
			return
		}
		s.writeBatchError(w, s.internalBatchError(err, "file_id", f.ID))
		return
	}
	w.Header().Set("Content-Type", "application/jsonl")
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(content); err != nil {
		s.logger.Warn("batch: writing file content to the consumer failed", "file_id", f.ID, "error", err)
	}
}

// batchNotFound is the single 404 every ownership check funnels into, so a
// consumer cannot tell "belongs to someone else" from "does not exist".
func batchNotFound(kind string) *batchError {
	return &batchError{
		Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found",
		Message: fmt.Sprintf("no such %s", kind),
	}
}
