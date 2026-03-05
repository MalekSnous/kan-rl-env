# KAN RL Environment — convenience commands
# Usage: make run | make build | make clean | make judge | make logs

.PHONY: run build rebuild clean judge logs stop prune

# ── Main commands ─────────────────────────────────────────────────────────────

# Always stop+remove the old container before starting fresh
run:
	docker compose --env-file .env down --remove-orphans 2>/dev/null || true
	docker compose --env-file .env up --force-recreate rl-env

# Restart without rebuild — uses mounted volumes, no reinstall
restart:
	docker compose --env-file .env down --remove-orphans 2>/dev/null || true
	docker compose --env-file .env up --force-recreate rl-env

# Build without cache
build:
	docker compose build --no-cache rl-env

# Build + run in one shot
rebuild:
	$(MAKE) build && $(MAKE) run

# Run judge only on existing solution
judge:
	docker compose --env-file .env run --rm --profile tools judge

# Tail logs of a running container
logs:
	docker compose logs -f rl-env

# Stop container
stop:
	docker compose --env-file .env down --remove-orphans

# Remove dangling images (safe — keeps latest)
prune:
	docker image prune -f
	@echo "Current images:"
	@docker images kan-rl-env

# Full cleanup: stop + prune
clean: stop prune