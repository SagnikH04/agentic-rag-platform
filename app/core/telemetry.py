from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.core.config import settings

_configured = False


def configure_telemetry() -> None:
    global _configured
    if _configured:
        return

    provider = TracerProvider(resource=Resource.create({"service.name": settings.app_name}))
    exporter = (
        OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
        if settings.otel_exporter_otlp_endpoint
        else ConsoleSpanExporter()
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True


def setup_telemetry(app) -> None:
    configure_telemetry()
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health/live,health/ready")


tracer = trace.get_tracer(settings.app_name)
