"""Module containing base classes for all distributions to inherit from.
"""
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node
from abc import abstractmethod
from jax._src.typing import ArrayLike, Array

###############################################################################
# Record of implemented distributions
###############################################################################
@dataclass(frozen=True)
class DistMap:
    continuous_names: tuple # tuple of continuous univariate distribution names
    discrete_names: tuple # tuple of discrete univariate distribution names
    mvt_names: tuple
    copula_names: tuple

    univariate_names: tuple = field(init=False)

    ids: tuple = field(init=False)  # tuple of all distribution ids
    names: tuple = field(init=False)  # tuple of all distribution names
    dtypes: tuple = field(init=False)  # tuple of all distribution data types
    dist_types: tuple = field(init=False)  # tuple of all distribution types
    object_names: tuple = field(init=False)  # tuple of all distribution object names

    id_map: dict = field(init=False)  # dict mapping distribution ids to names and dist_type
    name_map: dict = field(init=False)  # dict mapping distribution names to ids and dist_type
    object_name_map: dict = field(init=False)  # dict mapping distribution object names to ids and dist_type

    def __post_init__(self):
        univariate_names: tuple = self.continuous_names + self.discrete_names

        id_map: dict = {}
        name_map: dict = {}
        object_name_map: dict = {}

        d: dict = {'univariate': univariate_names, 'multivariate': self.mvt_names, 'copula': self.copula_names}
        start: int = 0
        for s, names in d.items():
            for i, name in enumerate(names, start=start):
                dtype: str = 'discrete' if name in discrete_names else 'continuous'
                object_name: str = name.replace('-', '_').lower()
                entry: dict = {'id': i, 'name': name, 'object_name': object_name, 'dtype': dtype, 'dist_type': s}
                id_map[i] = entry
                name_map[name] = entry
                object_name_map[object_name] = entry
            start = i + 1

        super().__setattr__('univariate_names', univariate_names)
        super().__setattr__('ids', tuple(id_map.keys()))
        super().__setattr__('names', tuple(name_map.keys()))
        super().__setattr__('object_names', tuple(object_name_map.keys()))
        super().__setattr__('dtypes', ('continuous', 'discrete'))
        super().__setattr__('dist_types', tuple(d.keys()))
        super().__setattr__('id_map', id_map)
        super().__setattr__('name_map', name_map)
        super().__setattr__('object_name_map', object_name_map)



continuous_names: tuple = ("Uniform", "Normal", "LogNormal", "Student-T", 
                           "Gamma", "Skewed-T", "GIG", "GH", "IG")
discrete_names: tuple = ()
mvt_names: tuple = ()
copula_names: tuple = ()

dist_map = DistMap(continuous_names=continuous_names, 
                   discrete_names=discrete_names, mvt_names=mvt_names, 
                   copula_names=copula_names)


###############################################################################
# Distribution PyTree / base class
###############################################################################
class Distribution:
    r"""Base class for all implemented copulAX distributions."""
    def __init__(self, name: str):
        self._id: int = dist_map.name_map[name]['id']

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, values: tuple, **init_kwargs):
        id_: int = aux_data[0]
        return cls(dist_map.id_map[id_]['name'], **init_kwargs)
    
    def tree_flatten(self):
        children = ()  # arrays and pytrees
        aux_data = (self._id,)  # static, hashable data
        return children, aux_data
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree_node(cls, cls.tree_flatten, cls.tree_unflatten)

    @property
    def name(self) -> str:
        """The name of the distribution.
        
        Returns:
            str: The name of the distribution.
        """
        return dist_map.id_map[self._id]['name']
    
    @property
    def dist_type(self) -> str:
        """The type of copulAX distribution.
        
        Returns:
            str: The type of the distribution.
        """
        return dist_map.id_map[self._id]['dist_type']
    
    @property
    def dtype(self) -> str:
        """The data type of the distribution.
        
        Returns:
            str: The data type of the distribution.
        """
        return dist_map.id_map[self._id]['dtype']
    
    @abstractmethod
    def _args_transform(self, *args, **kwargs) -> tuple:
        r"""Transforms the input arguments to the correct dtype and shape.
        """
    
    @abstractmethod
    def _params_dict(self, *args, **kwargs) -> dict:
        r"""Returns a dictionary of the distribution parameters.
        """
    
    @abstractmethod
    def fit(self, x: ArrayLike, *args, **kwargs):
        """Fit the distribution to the input data.
        
        Args:
            x (ArrayLike): The input data to fit the distribution to.
            kwargs: Additional keyword arguments to pass to the fit method.
        
        Returns:
            dict: The fitted distribution parameters.
        """
    
    @abstractmethod
    def rvs(self, *args, **kwargs) -> Array:
        """Generate random variates from the distribution.
        
        Args:
            kwargs: Additional keyword arguments to pass to the rvs method.
        
        Returns:
            jnp.ndarray: The generated random variates.
        """

    def sample(self, *args, **kwargs) -> Array:
        """Alias for the rvs method."""
        return self.rvs(*args, **kwargs)
    