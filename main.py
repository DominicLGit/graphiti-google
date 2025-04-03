import os
from datetime import UTC, datetime

from anyio import run
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from google_adapters.cross_encoder_client import GeminiRerankerClient
from google_adapters.embedder import GeminiEmbedderClient, GeminiEmbedderConfig
from google_adapters.llm_client import GeminiClient, LLMConfig

api_key = os.getenv("GOOGLE_API_KEY")


llm_config = LLMConfig(
    api_key=api_key,
)
embedder_config = GeminiEmbedderConfig(
    api_key=api_key,
)


async def main():
    graphiti = Graphiti(
        "bolt://localhost:7687",
        "neo4j",
        "password",
        llm_client=GeminiClient(llm_config),
        embedder=GeminiEmbedderClient(embedder_config),
        cross_encoder=GeminiRerankerClient(llm_config),
    )
    await clear_data(graphiti.driver)

    await graphiti.build_indices_and_constraints()

    episodes = [
        "Max loves donuts",
        "Max is a dog",
        "Max hates cats",
        "Max hates donutsBob is a cat",
        "cats hate Max",
    ]
    for i, episode in enumerate(episodes):
        await graphiti.add_episode(
            name=f"Freakonomics Radio {i}",
            episode_body=episode,
            source=EpisodeType.text,
            source_description="podcast",
            reference_time=datetime.now(tz=UTC),
        )


run(main)
