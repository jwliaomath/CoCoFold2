"""Validate the experiment's projection frame before model construction."""

import numpy as np


def resolve_projection_settings(args):
    frame = getattr(args, 'projection_frame', 'legacy')
    if frame not in ('legacy', 'fixed'):
        raise ValueError('--projection-frame must be legacy or fixed')
    origin = getattr(args, 'projection_origin', None)
    if origin is None:
        if frame == 'fixed':
            raise ValueError('--projection-origin X_A Y_A Z_A is required for fixed-frame projection; '
                             'choose a point in the placed CIF/map coordinate frame')
        origin = (0., 0., 0.)
    if len(origin) != 3 or not np.isfinite(origin).all():
        raise ValueError('--projection-origin requires three finite Angstrom coordinates')
    if frame == 'legacy' and tuple(origin) != (0., 0., 0.):
        raise ValueError('--projection-origin requires --projection-frame fixed')
    args.projection_origin = tuple(float(value) for value in origin)
