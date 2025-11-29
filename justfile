set shell := ["bash", "-uc"]

# Default recipe: show help
default:
    @just --list

# ============================================================================
# Development
# ============================================================================

# Run all checks (format, clippy, test, doc)
all: fmt-check clippy test doc-check

# Format code
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Run clippy with all features and targets
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Run clippy and auto-fix issues
clippy-fix:
    cargo clippy --all-targets --all-features --fix --allow-dirty --allow-staged

# ============================================================================
# Building & Checking
# ============================================================================

# Check compilation for all feature combinations
check:
    cargo check --all-targets --all-features

# Check no_std compatibility
check-no-std:
    cargo check --no-default-features
    cargo check --no-default-features --features serde

# Check all feature combinations
check-features:
    cargo check
    cargo check --no-default-features
    cargo check --no-default-features --features serde
    cargo check --features parallel
    cargo check --features serde
    cargo check --all-features

# Build in release mode
build:
    cargo build --release --all-features

# Clean build artifacts
clean:
    cargo clean

# ============================================================================
# Testing
# ============================================================================

# Run all tests
test:
    cargo test --all-features

# Run tests with all feature combinations (like CI)
test-all:
    cargo test
    cargo test --no-default-features
    cargo test --no-default-features --features serde
    cargo test --features parallel
    cargo test --features serde
    cargo test --all-features

# Run tests with output shown
test-verbose:
    cargo test --all-features -- --nocapture

# Run a specific test
test-one name:
    cargo test --all-features {{name}} -- --nocapture

# Run benchmarks
bench:
    cargo bench

# Run benchmarks and save baseline
bench-save name:
    cargo bench -- --save-baseline {{name}}

# Compare benchmarks against baseline
bench-compare name:
    cargo bench -- --baseline {{name}}

# ============================================================================
# Documentation
# ============================================================================

# Build documentation
doc:
    cargo doc --no-deps --all-features

# Build and open documentation
doc-open:
    cargo doc --no-deps --all-features --open

# Check documentation for warnings
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# ============================================================================
# Release
# ============================================================================

# Prepare release: run all checks
release-check: all check-features check-no-std test-all
    @echo "✓ All release checks passed!"

# Dry-run publish to crates.io
publish-dry:
    cargo publish --dry-run --all-features

# Bump version (patch), commit, and tag
release-patch: release-check
    #!/usr/bin/env bash
    set -euo pipefail

    # Get current version
    current=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')

    # Calculate new version (patch bump)
    IFS='.' read -r major minor patch <<< "$current"
    new_version="$major.$minor.$((patch + 1))"

    echo "Bumping version: $current -> $new_version"

    # Update Cargo.toml
    sed -i '' "s/^version = \"$current\"/version = \"$new_version\"/" Cargo.toml

    # Update Cargo.lock
    cargo check --quiet

    # Commit and tag
    git add Cargo.toml Cargo.lock
    git commit -m "release: v$new_version"
    git tag -a "v$new_version" -m "Release v$new_version"

    echo "✓ Released v$new_version"
    echo "Run 'git push && git push --tags' to publish"

# Bump version (minor), commit, and tag
release-minor: release-check
    #!/usr/bin/env bash
    set -euo pipefail

    current=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
    IFS='.' read -r major minor patch <<< "$current"
    new_version="$major.$((minor + 1)).0"

    echo "Bumping version: $current -> $new_version"

    sed -i '' "s/^version = \"$current\"/version = \"$new_version\"/" Cargo.toml
    cargo check --quiet

    git add Cargo.toml Cargo.lock
    git commit -m "release: v$new_version"
    git tag -a "v$new_version" -m "Release v$new_version"

    echo "✓ Released v$new_version"
    echo "Run 'git push && git push --tags' to publish"

# Bump version (major), commit, and tag
release-major: release-check
    #!/usr/bin/env bash
    set -euo pipefail

    current=$(grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
    IFS='.' read -r major minor patch <<< "$current"
    new_version="$((major + 1)).0.0"

    echo "Bumping version: $current -> $new_version"

    sed -i '' "s/^version = \"$current\"/version = \"$new_version\"/" Cargo.toml
    cargo check --quiet

    git add Cargo.toml Cargo.lock
    git commit -m "release: v$new_version"
    git tag -a "v$new_version" -m "Release v$new_version"

    echo "✓ Released v$new_version"
    echo "Run 'git push && git push --tags' to publish"

# Publish to crates.io (after release-*)
publish:
    cargo publish --all-features

# ============================================================================
# Utilities
# ============================================================================

# Show outdated dependencies
outdated:
    cargo outdated

# Update dependencies
update:
    cargo update

# Run security audit
audit:
    cargo audit

# Show dependency tree
tree:
    cargo tree

# Count lines of code
loc:
    @tokei --exclude target

# Watch for changes and run tests
watch:
    cargo watch -x 'test --all-features'

# Watch for changes and run clippy
watch-clippy:
    cargo watch -x 'clippy --all-features'
