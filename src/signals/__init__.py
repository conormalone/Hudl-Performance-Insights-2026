"""New signal package for player/team-level cognitive load measures.

Contains signals that augment the existing model2 framework with
per-player and per-team metrics that use a 'stacked' output schema
(game_id, block_id, phase, player_id, team_id_opta, signal_name,
signal_value, n_frames).

Signals are registered via the @register_signal decorator.
"""
