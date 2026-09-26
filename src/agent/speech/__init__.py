"""Speech in and out of the agent: audio helpers, text preparation for TTS, STT/TTS clients.

Everything here works on the telephone format the rest of the app speaks: mono 16-bit PCM at
8 kHz (what Asterisk's AudioSocket carries). Audio from other sources is converted at the edge.
"""
