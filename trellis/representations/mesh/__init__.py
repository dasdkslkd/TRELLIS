try:
	from .cube2mesh import SparseFeatures2Mesh, MeshExtractResult
except ImportError as exc:
	class MeshExtractResult:
		pass

	class SparseFeatures2Mesh:
		def __init__(self, *args, **kwargs):
			raise ImportError('flexicubes extension is unavailable') from exc
