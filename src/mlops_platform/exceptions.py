class PlatformError(Exception):
    """Base error for the serving platform."""


class DataValidationError(PlatformError):
    pass


class QualityGateError(PlatformError):
    pass


class RegistryError(PlatformError):
    pass


class ModelNotReadyError(PlatformError):
    pass


class FeatureSchemaError(PlatformError):
    pass
