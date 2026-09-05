package testbed

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/eigeninference/d-inference/coordinator/api"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// DefaultTestModelID is the checkpoint the testbed serves by default.
//
// v0.7.5 ONE-ENGINE: the provider serves exclusively through
// ContinuousBatchingV2 and never advertises models without a CBv2 adapter,
// so the old tiny-Qwen fixture (no adapter) became unservable BY DESIGN.
// gpt-oss-20b is the smallest CBv2-supported production checkpoint
// (~12 GB weights — the runner needs it in the HF cache). Override with
// DARKBLOOM_TESTBED_MODEL for machines that cache a different supported
// checkpoint.
func DefaultTestModelID() string {
	if m := os.Getenv("DARKBLOOM_TESTBED_MODEL"); m != "" {
		return m
	}
	return "mlx-community/gpt-oss-20b-MXFP4-Q8"
}

// SecondaryTestModelID is the second checkpoint multi-model suites serve.
// It must also be CBv2-servable (gpt_oss / gemma4 model families only —
// the provider filters advertised models through EngineV2SupportedModels,
// so a non-CBv2 checkpoint here would never register and its requests
// would only measure routing failures). Override with
// DARKBLOOM_TESTBED_MODEL_B.
func SecondaryTestModelID() string {
	if m := os.Getenv("DARKBLOOM_TESTBED_MODEL_B"); m != "" {
		return m
	}
	return "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
}

type ModelSpec struct {
	// ModelID is the single-model shorthand. ModelIDs takes precedence when set.
	ModelID string
	// ModelIDs lets one provider process advertise multiple models.
	ModelIDs     []string
	NumProviders int
}

func (ms ModelSpec) IDs() []string {
	if len(ms.ModelIDs) > 0 {
		return ms.ModelIDs
	}
	if ms.ModelID != "" {
		return []string{ms.ModelID}
	}
	return nil
}

var KnownModelSizes = map[string]string{
	"mlx-community/gpt-oss-20b-MXFP4-Q8":        "12.1 GB",
	"mlx-community/gemma-4-26B-A4B-it-qat-4bit": "14.5 GB",
	"mlx-community/Qwen3.5-0.8B-MLX-4bit":       "0.5 GB",
	"mlx-community/gemma-3-270m-4bit":           "0.2 GB",
}

type TrustLevel string

const (
	TrustNone       TrustLevel = "none"
	TrustSelfSigned TrustLevel = "self_signed"
	TrustHardware   TrustLevel = "hardware"
)

// KVBackendAuto / KVBackendPaged / KVBackendContiguous are the values the
// provider accepts for `engine_v2_kv_backend` under `[backend]`.
const (
	KVBackendAuto       = "auto"
	KVBackendPaged      = "paged"
	KVBackendContiguous = "contiguous"
)

// ProductionFirstContentDeadlineBase is the production fixed term exercised by
// E2E. It mirrors EIGENINFERENCE_TTFT_LIVE_DEADLINE_BASE_MS=9000; the
// coordinator adds 1ms per estimated prompt token.
const ProductionFirstContentDeadlineBase = 9 * time.Second

// ResolveKVBackend returns the KV backend the testbed should ask the provider
// for: the explicit value when set, else DARKBLOOM_TESTBED_KV_BACKEND, else ""
// (leave the provider at its own default). The env fallback lets CI re-run the
// existing suites against a different backend without editing a single test.
func ResolveKVBackend(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return os.Getenv("DARKBLOOM_TESTBED_KV_BACKEND")
}

// DevRestartMnemonic is the fixed, publicly-known BIP39 all-zero test vector
// the dev stack falls back to when it is asked to keep a persistent database
// across restarts and the environment names no mnemonic of its own.
//
// It is NOT a secret and must never reach a deployment: it exists so
// `make dev-stack --postgres` against EIGENINFERENCE_DATABASE_URL takes the
// PRODUCTION key path (sealedblob.DeriveKey) instead of
// EIGENINFERENCE_BATCH_DEV_INSECURE_KEY's process-local random key. A random
// key makes every blob written by the previous process undecryptable, which
// is what stops a restarted dev stack from resuming a batch it had already
// sealed inputs for.
const DevRestartMnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"

// ResolveEncryptionMnemonic returns the BIP39 mnemonic the suite's coordinator
// derives its sender-encryption and batch-store keys from: the explicit value
// when set, else the same MNEMONIC / EIGENINFERENCE_MNEMONIC pair
// billing.ReadConfig reads in production, else "".
//
// "" is not a failure: api.NewBatchBlobStore falls back to
// EIGENINFERENCE_BATCH_DEV_INSECURE_KEY (a process-local random key) and
// otherwise leaves the batch lane off.
func ResolveEncryptionMnemonic(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if m := os.Getenv("MNEMONIC"); m != "" {
		return m
	}
	return os.Getenv("EIGENINFERENCE_MNEMONIC")
}

// ResolveMaxConcurrent returns the per-slot concurrency cap: the explicit
// value when non-zero, else DARKBLOOM_TESTBED_MAX_CONCURRENT, else 0 (leave
// the provider to pick).
//
// 0 IS NOT "8". It is the provider's default, which is 4 as of v0.8.1 — the
// release that reverted v0.8.0's raise to 8 along with the paged KV default
// that justified it. Both the memberwise init and the config-decoder fallback
// read one constant (`BackendSettings.defaultEngineV2MaxConcurrent`, pinned
// together by maxConcurrentMemberwiseAndDecodeDefaultsCannotDrift), so unlike
// the pre-v0.8.0 split it no longer matters whether a TOML was written.
//
// Paged at B=4 measures 0.98x of contiguous against 1.17x at B=8, so selecting
// a KV backend and leaving this at 0 is the one configuration with all of
// paged's cost and none of its benefit. A lane that wants paged MUST name the
// cap too.
//
// A malformed env value is a hard error rather than a silent fall-through. It
// used to be ignored "over a typo in an optional knob", and the knob stopped
// being optional when it became the difference between measuring paged and
// measuring nothing: a typo would quietly seat the suite at the default and
// still pass.
func ResolveMaxConcurrent(explicit int) (int, error) {
	if explicit != 0 {
		return explicit, nil
	}
	raw := os.Getenv("DARKBLOOM_TESTBED_MAX_CONCURRENT")
	if raw == "" {
		return 0, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf(
			"DARKBLOOM_TESTBED_MAX_CONCURRENT=%q is not an integer: %w", raw, err)
	}
	return n, nil
}

// DescribeKVPosture renders the KV posture this config resolves to, naming the
// PROVENANCE of each value and not just the value. A run that reads back
// "provider default" in its log is a run nobody chose the backend for.
//
// This describes what the testbed ASKS the provider for. It equals what the
// provider BUILDS only because an explicit "paged" refuses rather than degrades
// (EngineV2KVBackendPolicy.degradesPagedFailure is false for .paged), so a
// paged request that could not be honoured fails the run instead of reaching
// this log with a lie in it. Under "auto" — or under the negative-polarity kill
// switch DARKBLOOM_CBV2_PAGED_KV=0, which always degrades — the two can differ,
// and only the provider's own slot log says which one served.
func DescribeKVPosture(cfg ProviderConfig) string {
	backend := "provider default"
	switch {
	case cfg.KVBackend != "":
		backend = cfg.KVBackend + " (suite)"
	case os.Getenv("DARKBLOOM_TESTBED_KV_BACKEND") != "":
		backend = os.Getenv("DARKBLOOM_TESTBED_KV_BACKEND") +
			" (env DARKBLOOM_TESTBED_KV_BACKEND)"
	}

	concurrent := "provider default"
	switch {
	case cfg.MaxConcurrent != 0:
		concurrent = fmt.Sprintf("%d (suite)", cfg.MaxConcurrent)
	case os.Getenv("DARKBLOOM_TESTBED_MAX_CONCURRENT") != "":
		concurrent = os.Getenv("DARKBLOOM_TESTBED_MAX_CONCURRENT") +
			" (env DARKBLOOM_TESTBED_MAX_CONCURRENT)"
	}

	kill := "unset"
	if raw := os.Getenv("DARKBLOOM_CBV2_PAGED_KV"); raw != "" {
		kill = raw
	}

	return fmt.Sprintf(
		"kv_backend=%s max_concurrent=%s DARKBLOOM_CBV2_PAGED_KV=%s",
		backend, concurrent, kill)
}

type ProviderConfig struct {
	TrustLevel                 TrustLevel
	ModelID                    string
	ModelIDs                   []string
	AttestationInterval        time.Duration
	AuthTokenPath              string
	EnableEphemeralPrefixCache bool
	// KVBackend selects the CBv2 KV-cache backend for every engine slot the
	// provider builds: "" (leave the provider at its own default), "auto",
	// "paged", or "contiguous". Non-empty makes the testbed write a provider
	// TOML into StateDir and launch with `--config`; empty changes nothing
	// about the launch.
	//
	// GOTCHA — do NOT reach for DARKBLOOM_CBV2_PAGED_KV to turn paged on.
	// That env var is negative-polarity ONLY: it is the fleet kill switch and
	// can force paged OFF, never ON. `engine_v2_kv_backend` under `[backend]`
	// is the only way to select paged, which is the entire reason the testbed
	// writes a config file at all.
	//
	// SETTING THIS ALONE IS A TRAP. Leaving MaxConcurrent at 0 seats the
	// provider on its own default, which is 4 as of v0.8.1. So "paged" on its
	// own is paged@4: 0.98x of contiguous, against 1.17x at B=8. Name
	// MaxConcurrent whenever you name KVBackend.
	KVBackend string
	// MaxConcurrent is the box-wide concurrent-request cap per engine slot
	// (`engine_v2_max_concurrent` under `[backend]`). The provider clamps the
	// value to [1, 8]. Travels through the same generated TOML as KVBackend.
	//
	// 0 leaves the provider to pick, which is 4 as of v0.8.1 — see
	// ResolveMaxConcurrent for why 0 is not a way to ask for 8.
	MaxConcurrent int
	// MTPDrafterPath is an explicit immutable local assistant snapshot. Empty
	// preserves provider policy: exact-model automatic MTP may use catalog
	// metadata and otherwise falls back to target-only decode.
	MTPDrafterPath string
}

func DefaultProviderConfig() ProviderConfig {
	return ProviderConfig{
		TrustLevel:          TrustNone,
		AttestationInterval: 5 * time.Minute,
	}
}

type RequestConfig struct {
	PromptTokens  int
	MaxTokens     int
	Streaming     bool
	Temperature   float64
	Concurrency   int
	TotalRequests int
	ModelID       string
	PromptBytes   int
}

func DefaultRequestConfig() RequestConfig {
	return RequestConfig{
		PromptTokens:  64,
		MaxTokens:     128,
		Streaming:     true,
		Temperature:   0.0,
		Concurrency:   1,
		TotalRequests: 10,
	}
}

type TestConfig struct {
	Model    ModelConfig
	Provider ProviderConfig
	Request  RequestConfig
}

func DefaultTestConfig() TestConfig {
	return TestConfig{
		Model:    DefaultModelConfig(),
		Provider: DefaultProviderConfig(),
		Request:  DefaultRequestConfig(),
	}
}

type ModelConfig struct {
	ModelID            string
	Quantization       string
	BackendPort        int
	ContinuousBatching bool
}

func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		ModelID:     "mlx-community/gemma-3-270m",
		BackendPort: 8000,
	}
}

type UserAccount struct {
	AccountID string
	APIKey    string
}

type SuiteConfig struct {
	ModelSpecs                 []ModelSpec
	NumUsers                   int
	QueueCapacity              int
	QueueTimeout               time.Duration
	FirstContentDeadlineBase   time.Duration
	SeedBalance                int64
	UseMemoryStore             bool
	EnableEphemeralPrefixCache bool
	// CatalogModels and ModelAliases seed the suite's isolated DB before the
	// provider connects. Empty preserves the lightweight ID-only catalog used
	// by the ordinary testbed.
	CatalogModels []CatalogModel
	ModelAliases  []store.ModelAlias
	// ExpectedProviderCapabilities asks the testbed to admit only providers
	// that actually reported every capability at registration. It is opt-in;
	// ordinary suites retain the historical self-signed force-trust behavior.
	ExpectedProviderCapabilities []string
	// MTPDrafterPath is forwarded only when a test explicitly configured a
	// local immutable assistant snapshot.
	MTPDrafterPath string
	// KVBackend / MaxConcurrent are forwarded verbatim to every provider this
	// suite launches. See ProviderConfig for the zero-value semantics and the
	// DARKBLOOM_CBV2_PAGED_KV gotcha.
	KVBackend     string
	MaxConcurrent int
	// ExpectKVBackend asserts the KV backend every engine slot was actually
	// BUILT with ("paged" or "contiguous"); the suite pre-warms each slot and
	// fails Start when the heartbeat-reported kv_backend differs or never
	// arrives. "" falls back to DARKBLOOM_TESTBED_EXPECT_KV_BACKEND, and an
	// unset env leaves the assertion off. Orthogonal to KVBackend: that knob
	// REQUESTS a backend, this one asserts the CONSTRUCTED one — a lane
	// exercising the `.auto` default sets only the expectation.
	ExpectKVBackend string
	// DatabaseURL attaches a non-memory suite to an EXISTING Postgres
	// instead of provisioning an ephemeral one. Empty provisions the
	// ephemeral container/initdb the testbed owns and removes on Stop.
	//
	// The testbed does not create, migrate away or drop a database named
	// here — only Suite.Stop's ephemeral path removes anything — so a dev
	// stack restarted against the same URL sees the rows the previous
	// process left behind.
	//
	// DELIBERATELY has no environment fallback. The obvious candidate,
	// EIGENINFERENCE_DATABASE_URL, is WRITTEN by the testbed itself:
	// deps.PostgresLifecycle.SetEnv exports the ephemeral instance's URL into
	// the process. A suite that read it back would, on the second suite in a
	// test binary, attach to the first suite's already-removed database. Only
	// a caller that knows it owns a database sets this field —
	// e2e/cmd/devstack resolves the developer-facing variable itself.
	DatabaseURL string
	// APIKey asks the suite to reuse an already-issued raw key for Users[0]
	// instead of minting a fresh one. Empty always mints.
	//
	// The store generates its own key material and stores only a hash, so a
	// key that is not already present cannot be created from a caller-supplied
	// secret: createUserPool adopts the key when the store validates it and
	// otherwise WARNs and mints. That makes this useful for exactly one thing
	// — carrying the key a previous run printed across a restart against a
	// persistent DatabaseURL.
	APIKey string
	// EncryptionMnemonic is forwarded to billing.Config so the coordinator
	// derives its sender-encryption key and the batch lane's seal key from it,
	// exactly as coordinator/cmd/coordinator/main.go does. Empty falls back to
	// MNEMONIC / EIGENINFERENCE_MNEMONIC (see ResolveEncryptionMnemonic).
	EncryptionMnemonic string
	// ListenAddr pins the coordinator's HTTP listener to a fixed address
	// (e.g. "127.0.0.1:18080") instead of the ephemeral 127.0.0.1:0 default.
	// Empty preserves the ephemeral-port behaviour every existing suite
	// relies on.
	ListenAddr string
	// RealChallenges runs the coordinator's production attestation-challenge
	// loop: challenges are actually sent on the coordinator's default interval
	// and the provider must answer them. Default (false) skips challenges, the
	// posture every e2e suite has always used.
	//
	// Skipping is safe for a test that finishes in minutes and the coordinator
	// now keeps challenge freshness alive while it skips, so a long-lived stack
	// no longer derouts. This knob is about FIDELITY: set it when the lane
	// should exercise the real challenge/response path end to end
	// (e2e/cmd/devstack --real-challenges).
	RealChallenges bool
}

// challengePosture returns the coordinator's attestation-challenge settings for
// this suite: whether to skip challenges, and the challenge interval to
// configure. The default posture parks the interval at an hour because nothing
// is sent anyway; RealChallenges takes the production path on the coordinator's
// own default interval.
//
// Split out of startCoordinator so the plumbing is unit-testable without
// booting a coordinator.
func (sc SuiteConfig) challengePosture() (skip bool, interval time.Duration) {
	if sc.RealChallenges {
		return false, api.DefaultChallengeInterval
	}
	return true, 1 * time.Hour
}

func DefaultSuiteConfig() SuiteConfig {
	return SuiteConfig{
		ModelSpecs:               []ModelSpec{{ModelID: DefaultTestModelID(), NumProviders: 1}},
		NumUsers:                 1,
		QueueCapacity:            100,
		QueueTimeout:             120 * time.Second,
		FirstContentDeadlineBase: ProductionFirstContentDeadlineBase,
		SeedBalance:              100_000_000,
	}
}

func (sc SuiteConfig) AllModelIDs() []string {
	seen := make(map[string]bool)
	var ids []string
	for _, spec := range sc.ModelSpecs {
		for _, id := range spec.IDs() {
			if !seen[id] {
				seen[id] = true
				ids = append(ids, id)
			}
		}
	}
	return ids
}

func (sc SuiteConfig) TotalProviders() int {
	total := 0
	for _, spec := range sc.ModelSpecs {
		total += spec.NumProviders
	}
	return total
}

func (sc SuiteConfig) PrimaryModelID() string {
	if len(sc.ModelSpecs) > 0 {
		ids := sc.ModelSpecs[0].IDs()
		if len(ids) > 0 {
			return ids[0]
		}
	}
	return DefaultTestModelID()
}
