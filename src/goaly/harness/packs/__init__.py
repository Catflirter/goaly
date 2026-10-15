"""The capability packs (spec §5.2): one per benchmark, and the only code RSI may edit.

Pack code imports only what the static allowlist admits (spec §8.5), and in every real run it
runs only in the solve container (spec §6.1). Its harness factory, outside the pack, reads the
pack's files and builds its policy.
"""
