import asyncio
import uuid

from aiohttp import web


async def start_session(request):
    """
    启动会话的 API 端点
    请求 JSON 格式：
    {}
    """
    # 自动分配一个 session_id
    session_id = str(uuid.uuid4())

    return web.json_response({'code': 0, 'message': 'Session started', 'session_id': session_id})


async def stop_session(request):
    """
    停止会话的 API 端点
    请求 JSON 格式：
    {
        "session_id": "unique_session_id"
    }
    """
    data = await request.json()
    session_id = data.get('session_id')
    if not session_id:
        return web.json_response({'code': 1, 'message': 'session_id is required'}, status=400)

    return web.json_response({'code': 0, 'message': 'Session stopped'})

async def main():
    appasync = web.Application()
    appasync.router.add_post('/api/start_session', start_session)
    appasync.router.add_post('/api/stop_session', stop_session)

    # 创建一个Runner，并监听8010端口
    runner = web.AppRunner(appasync)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8010)
    await site.start()

    print("Server started at http://0.0.0.0:8010")
    # 无限等待，让服务器保持运行
    await asyncio.Event().wait()


if __name__ == '__main__':
    asyncio.run(main())