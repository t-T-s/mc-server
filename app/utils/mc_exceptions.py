from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import traceback

async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "path_params": request.path_params,
            "query_params": dict(request.query_params)
        },
    )


async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "headers": dict(request.headers),
            "path_params": request.path_params,
            "query_params": dict(request.query_params),
            "error": str(exc),
            "traceback": traceback.format_exc().strip().split("\n")[-1],  # It is a security risk to include the
            # whole traceback!
            "method": request.method,
            "url": str(request.url),
        },
    )
