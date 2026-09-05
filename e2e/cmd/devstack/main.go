// Command devstack runs a coordinator and one provider on fixed, well-known
// addresses for manual testing on a single Mac. It is the same in-process
// testbed.Suite the e2e integration tests use — never a hand-rolled
// coordinator or provider launch — so the dev loop exercises exactly the
// lifecycle CI does.
//
// Usage:
//
//	cd e2e && GOTOOLCHAIN=auto go run ./cmd/devstack [flags]
//
// The coordinator listens on http://127.0.0.1:18080. Stop with Ctrl-C
// (SIGINT); the suite tears down the provider and coordinator cleanly.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/eigeninference/d-inference/e2e/testbed"
)

const devstackListenAddr = "127.0.0.1:18080"

func main() {
	usePostgres := flag.Bool("postgres", false, "use a real Postgres store (EIGENINFERENCE_DATABASE_URL) instead of the in-memory store")
	model := flag.String("model", "", "model ID to serve (default: DARKBLOOM_TESTBED_MODEL env, then the testbed default)")
	providerBinary := flag.String("provider-binary", "", "path to a prebuilt provider binary (default: DARKBLOOM_PROVIDER_BINARY env, else build from source via testbed.BuildProvider)")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if *providerBinary != "" {
		if err := os.Setenv("DARKBLOOM_PROVIDER_BINARY", *providerBinary); err != nil {
			logger.Error("set DARKBLOOM_PROVIDER_BINARY", "error", err)
			os.Exit(1)
		}
	}

	cfg := testbed.SuiteConfig{
		UseMemoryStore: !*usePostgres,
		ListenAddr:     devstackListenAddr,
	}
	if *model != "" {
		cfg.ModelSpecs = []testbed.ModelSpec{{ModelID: *model, NumProviders: 1}}
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	suite := testbed.NewSuite(cfg)
	logger.Info("starting dev stack", "listen_addr", devstackListenAddr, "postgres", *usePostgres, "model", cfg.PrimaryModelID())
	if err := suite.Start(ctx); err != nil {
		logger.Error("dev stack failed to start", "error", err)
		os.Exit(1)
	}

	fmt.Printf("\nDev stack ready.\n")
	fmt.Printf("  Base URL: %s\n", suite.Coordinator.BaseURL())
	if len(suite.Users) > 0 {
		fmt.Printf("  API key:  %s\n", suite.Users[0].APIKey)
	}
	for _, p := range suite.Providers {
		fmt.Printf("  Provider PID: %d (model: %s)\n", p.PID(), cfg.PrimaryModelID())
	}
	fmt.Printf("\nTo run the smoke script:\n")
	if len(suite.Users) > 0 {
		fmt.Printf("  DARKBLOOM_DEV_URL=%s DARKBLOOM_DEV_KEY=%s ./scripts/dev-smoke-chat.sh\n", suite.Coordinator.BaseURL(), suite.Users[0].APIKey)
	}
	fmt.Printf("\nPress Ctrl-C to stop.\n\n")

	<-ctx.Done()
	logger.Info("shutting down dev stack")
	suite.Stop()
	logger.Info("dev stack stopped")
}
