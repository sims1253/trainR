#!/bin/bash
# Version-pinned Claude Code CLI installer
# Usage: ./scripts/install_claude_cli.sh [VERSION]
#
# This script installs a specific version of Claude Code CLI for reproducibility.
# Versions can be found at: https://www.npmjs.com/package/@anthropic-ai/claude-code

set -euo pipefail

# Default version (can be overridden via argument or environment variable)
DEFAULT_VERSION="1.0.33"
VERSION="${1:-${CLAUDE_CLI_VERSION:-$DEFAULT_VERSION}}"

# Installation directory
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
TEMP_DIR=$(mktemp -d)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Check for existing installation
check_existing() {
    if command -v claude &> /dev/null; then
        local current_version
        current_version=$(claude --version 2>/dev/null || echo "unknown")
        log_info "Existing Claude CLI found: version $current_version"
        
        if [[ "$current_version" == *"$VERSION"* ]]; then
            log_info "Version $VERSION already installed. Nothing to do."
            exit 0
        fi
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check for Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is required but not installed."
        log_info "Install Node.js from https://nodejs.org/"
        exit 1
    fi
    
    # Check for npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is required but not installed."
        exit 1
    fi
    
    local node_version
    node_version=$(node --version)
    log_info "Node.js version: $node_version"
    
    local npm_version
    npm_version=$(npm --version)
    log_info "npm version: $npm_version"
}

# Install via npm (recommended method)
install_via_npm() {
    log_info "Installing Claude CLI version $VERSION via npm..."
    
    local npm_flags=""
    if [[ "$INSTALL_DIR" != "/usr/local/bin" ]]; then
        npm_flags="--prefix $INSTALL_DIR"
    fi
    
    # Install globally with specific version
    if npm install -g @anthropic-ai/claude-code@"$VERSION" $npm_flags; then
        log_info "Successfully installed via npm"
        return 0
    else
        log_warn "npm installation failed, trying alternative method..."
        return 1
    fi
}

# Alternative: Install from GitHub release tarball
install_from_release() {
    log_info "Installing Claude CLI version $VERSION from GitHub release..."
    
    local arch
    arch=$(uname -m)
    
    case "$arch" in
        x86_64)
            arch="x64"
            ;;
        aarch64|arm64)
            arch="arm64"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    local download_url="https://github.com/anthropics/claude-code/releases/download/v${VERSION}/claude-code-linux-${arch}.tar.gz"
    local tarball_path="$TEMP_DIR/claude.tar.gz"
    
    log_info "Downloading from: $download_url"
    
    if ! curl -fsSL "$download_url" -o "$tarball_path"; then
        log_error "Failed to download Claude CLI v$VERSION"
        log_info "Available versions: https://github.com/anthropics/claude-code/releases"
        exit 1
    fi
    
    # Verify download
    if [[ ! -s "$tarball_path" ]]; then
        log_error "Downloaded file is empty"
        exit 1
    fi
    
    # Extract and install
    log_info "Extracting to $INSTALL_DIR..."
    tar -xzf "$tarball_path" -C "$TEMP_DIR"
    
    # Find and copy binary
    local binary_path
    binary_path=$(find "$TEMP_DIR" -name "claude" -type f | head -n1)
    
    if [[ -z "$binary_path" ]]; then
        log_error "Could not find claude binary in tarball"
        exit 1
    fi
    
    # Need sudo for system directories
    if [[ "$INSTALL_DIR" == "/usr/local/bin" ]] && [[ $EUID -ne 0 ]]; then
        log_info "Installing to $INSTALL_DIR requires sudo..."
        sudo cp "$binary_path" "$INSTALL_DIR/claude"
        sudo chmod +x "$INSTALL_DIR/claude"
    else
        cp "$binary_path" "$INSTALL_DIR/claude"
        chmod +x "$INSTALL_DIR/claude"
    fi
    
    log_info "Successfully installed from release tarball"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    if ! command -v claude &> /dev/null; then
        log_error "Claude CLI not found in PATH"
        exit 1
    fi
    
    local installed_version
    installed_version=$(claude --version 2>/dev/null || echo "unknown")
    
    log_info "Installed Claude CLI version: $installed_version"
    
    # Verify version matches (if version string contains the version number)
    if [[ "$installed_version" == *"$VERSION"* ]] || [[ "$installed_version" == "unknown" ]]; then
        log_info "✓ Installation verified successfully"
    else
        log_warn "Version mismatch: expected $VERSION, got $installed_version"
    fi
}

# Main installation flow
main() {
    log_info "Claude Code CLI Installer"
    log_info "Target version: $VERSION"
    log_info "Install directory: $INSTALL_DIR"
    echo ""
    
    check_existing
    check_prerequisites
    echo ""
    
    # Try npm first, fall back to release tarball
    if ! install_via_npm; then
        install_from_release
    fi
    
    echo ""
    verify_installation
}

# Run main function
main "$@"
