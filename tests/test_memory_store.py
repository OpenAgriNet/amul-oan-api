"""Direct Qdrant read contracts; HTTP is mocked and embeddings are synthetic."""
import asyncio
import json
import uuid
from unittest.mock import AsyncMock

import httpx
import pytest
from app.services import memory_store as store

REF = '11111111-1111-4111-8111-111111111111'
OTHER = '22222222-2222-4222-8222-222222222222'


def point(ref=REF, **payload):
    text = 'first detail. second detail.'
    return {'id': ref, 'payload': {'farmer_id': 'one', 'type': 'memory', 'headline': 'A remembered event',
            'expanded': text, 'chunks': [{'number': 1, 'start': 0, 'end': 14}, {'number': 2, 'start': 14, 'end': len(text)}],
            'chunk_count': 2, **payload}}


@pytest.fixture
def reader():
    obj = store.MemoryStore(None, 'one')
    obj._enabled = True
    return obj


@pytest.mark.parametrize('value,expected', [(False, False), ('true', False), (True, True), (None, False)])
def test_stored_flag_requires_boolean_true(value, expected):
    reader = store.MemoryStore(None, 'one')
    reader._points = AsyncMock(return_value=[point(type='settings', memory_enabled=value)])
    assert asyncio.run(reader.enabled()) is expected


def test_unset_flag_defaults_on_but_failed_lookup_never_does():
    reader = store.MemoryStore(None, 'one')
    reader._points = AsyncMock(return_value=[])
    assert asyncio.run(reader.enabled()) is True
    reader._enabled = None
    reader._points = AsyncMock(side_effect=httpx.ConnectError('offline'))
    with pytest.raises(httpx.ConnectError):
        asyncio.run(reader.enabled())


def test_flag_off_prevents_every_read(reader):
    reader._enabled = False
    reader._qdrant = AsyncMock()
    for method, args in [(reader.profile, ()), (reader.keys, ()), (reader.search, ('word',)),
                         (reader.list_entries, ()), (reader.read, (REF,)), (reader.history, (REF,))]:
        with pytest.raises(PermissionError):
            asyncio.run(method(*args))
    reader._qdrant.assert_not_called()


def test_keys_are_farmer_scoped_and_hide_identifiers(reader):
    reader._qdrant = AsyncMock(return_value={'points': [point(metadata={'animal_id': 'secret', 'scheme': 'fodder'}), point(OTHER, farmer_id='other', metadata={'private_topic': 'other farmer'})]})
    reader._points = AsyncMock(return_value=[])
    result = asyncio.run(reader.keys())
    assert result['keys'] == [{'filter_path': 'metadata.scheme', 'description': None, 'values_for_this_farmer': ['fodder']}]
    assert reader._qdrant.call_args.args[1]['filter']['must'][0] == {'key': 'farmer_id', 'match': {'value': 'one'}}


def test_exact_list_filters_and_cursor_do_not_override_farmer(reader):
    reader._qdrant = AsyncMock(return_value={'points': [point(metadata={'animal_id': 'hidden', 'stage': 2})], 'next_page_offset': OTHER})
    out = asyncio.run(reader.list_entries(status=['open', 'pending'], filters={'fat': 4.5, 'stage': 2}, cursor=REF))
    body = reader._qdrant.call_args.args[1]
    assert body['offset'] == REF
    assert {'key': 'status', 'match': {'any': ['open', 'pending']}} in body['filter']['must']
    assert {'key': 'metadata.fat', 'range': {'gte': 4.5, 'lte': 4.5}} in body['filter']['must']
    assert out['next_cursor'] == OTHER
    assert 'hidden' not in json.dumps(out) and 'farmer_id' not in out['entries'][0]
    with pytest.raises(ValueError):
        asyncio.run(reader.list_entries(filters={'farmer_id': 'other'}))


def test_read_checks_owner_and_never_shortens_chunks(reader, monkeypatch):
    reader._points = AsyncMock(return_value=[point()])
    monkeypatch.setattr(store, 'READ_CHUNKS', 1)
    first = asyncio.run(reader.read(REF))
    second = asyncio.run(reader.read(REF, first['continuation']['chunk_ids']))
    assert first['chunks'][0]['text'] + second['chunks'][0]['text'] == point()['payload']['expanded']
    reader._points = AsyncMock(return_value=[point(farmer_id='other')])
    with pytest.raises(ValueError):
        asyncio.run(reader.read(REF))


def test_long_episode_continuation_stays_usable_and_covers_all_text(reader, monkeypatch):
    text = ''.join(f'Detail {n}. ' for n in range(60))
    chunks = [{'number': n + 1, 'start': n, 'end': n + 1} for n in range(len(text))]
    reader._points = AsyncMock(return_value=[point(expanded=text, chunks=chunks, chunk_count=len(chunks))])
    monkeypatch.setattr(store, 'READ_CHUNKS', 3)
    async def read_all():
        pages, selection = [], None
        while True:
            result = await reader.read(REF, selection)
            pages.extend(chunk['text'] for chunk in result['chunks'])
            continuation = result['continuation']
            if not continuation:
                return ''.join(pages)
            selection = continuation['chunk_ids']
            assert len(selection) <= 3
    assert asyncio.run(read_all()) == text


def test_history_is_bounded_and_does_not_follow_other_farmer(reader):
    reader._points = AsyncMock(return_value=[point(previous_version=REF)])
    reader._qdrant = AsyncMock(return_value={'points': []})
    out = asyncio.run(reader.history(REF))
    assert len(out['history']) == 1 and out['truncated']
    reader._points = AsyncMock(side_effect=[[point(previous_version=OTHER)], [point(OTHER, farmer_id='other')]])
    with pytest.raises(ValueError):
        asyncio.run(reader.history(REF))


def test_search_rejects_orphan_and_superseded_parents(reader):
    chunks = [{'id': str(uuid.uuid4()), 'payload': {'type': 'memory_chunk', 'farmer_id': 'one', 'episode_id': ref,
               'chunk_number': 1, 'start': 0, 'end': 14}} for ref in (REF, OTHER)]
    reader._embed = AsyncMock(return_value=[1, 0, 0])
    reader._qdrant = AsyncMock(return_value={'points': chunks})
    reader._points = AsyncMock(return_value=[point(superseded_at='2026-09-01')])
    assert asyncio.run(reader.search('detail', using='expanded'))['hits'] == []
    body = reader._qdrant.call_args.args[1]
    assert {'is_empty': {'key': 'superseded_at'}} in body['filter']['must']


def test_search_matches_saved_parent_chunk_offsets(reader):
    child = {'id': OTHER, 'payload': {'type': 'memory_chunk', 'farmer_id': 'one', 'episode_id': REF,
             'chunk_number': 2, 'start': 14, 'end': 999}}
    reader._embed = AsyncMock(return_value=[1, 0, 0])
    reader._qdrant = AsyncMock(return_value={'points': [child]})
    reader._points = AsyncMock(return_value=[point()])
    assert asyncio.run(reader.search('detail', using='expanded'))['hits'] == []
    child['payload']['end'] = len(point()['payload']['expanded'])
    out = asyncio.run(reader.search('detail', using='expanded'))
    assert out['hits'][0]['passage'] == point()['payload']['expanded'][14:]
    assert 'expanded' not in out['hits'][0]


def test_qdrant_key_never_reaches_marqo_and_embeddings_use_query_convention(monkeypatch):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'embeddings': [[1, 0, 0]]} if request.url.host == 'marqo.test' else {'result': []})
    monkeypatch.setattr(store, 'QDRANT_URL', 'https://qdrant.test')
    monkeypatch.setattr(store, 'QDRANT_API_KEY', 'test-key')
    monkeypatch.setattr(store, 'MARQO_URL', 'http://marqo.test')
    monkeypatch.setattr(store, 'EMBED_DIM', 3)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            reader = store.MemoryStore(client, 'one')
            await reader.enabled()
            await reader._embed('receipt')
    asyncio.run(run())
    assert calls[0].headers['api-key'] == 'test-key'
    assert 'api-key' not in calls[1].headers
    assert json.loads(calls[1].content)['content_type'] == 'query'


def test_search_rejects_changed_embedding_model(reader):
    reader._embed = AsyncMock(return_value=[1, 0, 0])
    reader._qdrant = AsyncMock(return_value={'points': [point(embed_model='different-model')]})
    with pytest.raises(ValueError, match='embedding models differ'):
        asyncio.run(reader.search('detail'))
