"""Prompt analysis and seed track dimension extraction."""

from backend.llm_client import get_llm_client
from backend.models import (
    AnalyzePromptResponse,
    AnalyzeTrackResponse,
    Dimension,
    GenreCount,
    DecadeCount,
    Track,
)
from backend.plex_client import get_plex_client


PROMPT_ANALYSIS_SYSTEM = """You are a music expert helping to create playlists from a user's music library.

Analyze the user's prompt and suggest appropriate filters (genres and decades) that would help find matching tracks.

Return a JSON object with:
- genres: Array of genre names that match the prompt (e.g., ["Alternative", "Rock", "Indie"])
- decades: Array of decade strings (e.g., ["1990s", "2000s"])
- reasoning: Brief explanation of why you chose these filters

Be specific about genres and decades. Consider:
- Mood/atmosphere (melancholy, upbeat, energetic)
- Era references (90s, classic, modern)
- Genre keywords (alternative, jazz, electronic)
- Artist style hints

Return ONLY valid JSON, no markdown formatting."""


TRACK_ANALYSIS_SYSTEM = """You are a musicologist analyzing a track to extract its core musical anchors. 
Your goal is to identify 3–5 "Core Musical Anchors" that define this track's identity. These anchors must be grounded in the track's metadata and identifiable musical characteristics. 
STRICT PROHIBITION: Do not use arbitrary linguistic patterns, shared word endings, or purely subjective adjectives that lack musical meaning (e.g., do not use "songs that end in 's'" or "songs with 'the' in the title").
Focus on these anchor types:
1. Entity Anchor: The specific artist, album, or era (e.g., 'Post Malone', 'Midwest Emo era', '90s Grunge').
2. Genre Anchor: The precise sub-genre or niche (e.g., 'Dark Americana', 'Phonk', 'Melodic Trap').
3. Sonic/Thematic Anchor: The identifiable musical signature or lyrical theme (e.g., 'heavy 808s and auto-tune', 'acoustic guitar-driven', 'themes of heartbreak').
For each anchor, provide:
- id: A machine-readable identifier (e.g., "artist_identity", "sub_genre", "sonic_signature", "theme").
- label: A specific, descriptive name (e.g., "Post Malone", "Melodic Trap").
- description: A brief explanation of how this anchor defines this specific track.
Return a JSON object with the following structure:
{
  "dimensions": [
    {"id": "artist_identity", "label": "Post Malone", "description": "The specific melodic trap style of the artist."},
    {"id": "sub_genre", "label": "Melodic Trap", "description": "A blend of trap beats with pop-oriented melodies."},
    ...
  ]
}
Return ONLY valid JSON, no markdown formatting."""

def analyze_prompt(prompt: str) -> AnalyzePromptResponse:
    """Analyze a natural language prompt to suggest filters.

    Args:
        prompt: User's playlist description

    Returns:
        AnalyzePromptResponse with suggested and available filters

    Raises:
        ValueError: If LLM response cannot be parsed
        RuntimeError: If clients are not initialized
    """
    llm_client = get_llm_client()
    plex_client = get_plex_client()

    if not llm_client:
        raise RuntimeError("LLM client not initialized")
    if not plex_client:
        raise RuntimeError("Plex client not initialized")

    # Get library stats for available filters
    stats = plex_client.get_library_stats()
    available_genres = [GenreCount(**g) for g in stats.get("genres", [])]
    available_decades = [DecadeCount(**d) for d in stats.get("decades", [])]

    # Build prompt with available filter context
    analysis_prompt = f"""User's playlist request: "{prompt}"

Available genres in their library:
{', '.join(f"{g.name} ({g.count})" if g.count else g.name for g in available_genres[:30])}

Available decades in their library:
{', '.join(f"{d.name} ({d.count})" if d.count else d.name for d in available_decades)}

Suggest genres and decades from the available options that best match the user's request."""

    # Call LLM
    response = llm_client.analyze(analysis_prompt, PROMPT_ANALYSIS_SYSTEM)

    # Parse response
    try:
        data = llm_client.parse_json_response(response)
    except ValueError as e:
        raise ValueError(f"Failed to parse LLM response JSON: {e}")

    # Ensure genres and decades are lists to prevent TypeError during iteration
    genres = data.get("genres")
    decades = data.get("decades")
    genres = genres if isinstance(genres, list) else []
    decades = decades if isinstance(decades, list) else []

    # Filter suggestions to only include available options
    available_genre_names = {g.name for g in available_genres}
    available_decade_names = {d.name for d in available_decades}

    suggested_genres = [
        g for g in genres
        if g in available_genre_names
    ]
    suggested_decades = [
        d for d in decades
        if d in available_decade_names
    ]

    return AnalyzePromptResponse(
        suggested_genres=suggested_genres,
        suggested_decades=suggested_decades,
        available_genres=available_genres,
        available_decades=available_decades,
        reasoning=data.get("reasoning", ""),
        token_count=response.total_tokens,
        estimated_cost=response.estimated_cost(),
    )


def analyze_track(track: Track) -> AnalyzeTrackResponse:
    """Analyze a seed track to extract musical dimensions.

    Args:
        track: Track to analyze

    Returns:
        AnalyzeTrackResponse with track and dimensions

    Raises:
        ValueError: If LLM response cannot be parsed
        RuntimeError: If LLM client is not initialized
    """
    llm_client = get_llm_client()

    if not llm_client:
        raise RuntimeError("LLM client not initialized")

    # Build analysis prompt
    analysis_prompt = f"""Analyze this track:
Title: {track.title}
Artist: {track.artist}
Album: {track.album}
Year: {track.year or "Unknown"}
Genres: {", ".join(track.genres) if track.genres else "Unknown"}

Identify musical anchors that make this track distinctive."""

    # Call LLM
    response = llm_client.analyze(analysis_prompt, TRACK_ANALYSIS_SYSTEM)

    # Parse response
    try:
        data = llm_client.parse_json_response(response)
    except ValueError as e:
        raise ValueError(f"Failed to parse LLM response JSON: {e}")

    # Ensure dimensions is a list
    dimensions_data = data.get("dimensions")
    dimensions_data = dimensions_data if isinstance(dimensions_data, list) else []

    dimensions = [
        Dimension(
            id=d.get("id") or d.get("label", "Unknown dimension").lower().replace(" ", "_"),
            label=d.get("label", "Unknown dimension"),
            description=d.get("description", ""),
        )
        for d in dimensions_data
    ]

    return AnalyzeTrackResponse(
        track=track,
        dimensions=dimensions,
        token_count=response.total_tokens,
        estimated_cost=response.estimated_cost(),
    )
