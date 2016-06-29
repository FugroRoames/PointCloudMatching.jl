module PointCloudMatching

using PointClouds
using Displaz

export find_closest_points, register


"""
Find index of closest point in `ref` for every point in `query`

Args:
    query  - 3xN array of points used to query for nearest neighbours
    ref    - Points queryable with knn

Returns:
    Array of indices of the nearest neighbours for every query point
"""
function find_closest_points(query_points, ref)
    N = size(query_points,2)
    matchinds = zeros(Int, N)
    for i = 1:N
        p = query_points[:,i]
        neighbours, _ = knn(ref, p, 1)
        matchinds[i] = neighbours[1]
    end
    return matchinds
end


"""
Find the minimum value of the inlier fraction f in the FRMSD objective.

FRMSD = Fractional Root Mean Square Distance.

Ref: "Outlier Robust ICP for Minimizing Fractional RMSD" by Phillips, Liu and
Tomasi, http://arxiv.org/abs/cs/0606098

Args:
    d      - Scalar point matching residuals

Keyword args:
    lambda - Regularization tuning parameter from data inlier/outlier model.
             This depends heavily on the effective dimensionality, in
             particular it's quite different for point-to-point and
             point-to-surface matching.  The default value of lambda=0.95
             is the value recommended in Phillips et al. for 3D point clouds.

Returns:
    The indicies of the inliers and the associated value of f.
"""
function frmsd_inliers(d; lambda = 0.95)
    p = sortperm(d)  # sort indices
    N = length(d)
    f = (1:N)/N
    # Minimization of FRMSD as a function of inlier fraction by brute force.
    # This is quite efficient since it's a 1D problem and O(N) after sorting,
    frmsd = f.^-lambda .* sqrt(cumsum(d[p])./(1:N))
    last_inlier_ind = indmin(frmsd)
    return p[1:last_inlier_ind], f[last_inlier_ind]
end


"""
    register(cloud, ref_cloud; abstol::Float64=1e-3, dist_type::Symbol=:surface, maxit::Int=100, init_translation::Vector{Float64}=zeros(3), iterfunc=nothing)

Find optimum translation vector to register two point clouds.

Args:
    cloud      - Cloud of points to translate
    ref_cloud  - Cloud of points to serve as reference

Keyword args:
    abstol     - absolute tolerance required to stop the update step
    dist_type  - Method for computing residuals (:surface or :point)
    maxit      - Maximum iterations in ICP loop
    init_translation - Initial translation hint for where to start the
                 minimization.

Returns:
    Translation vector and bitarray with inlier flags
"""
function register(cloud, ref_cloud; abstol::Float64=1e-3, dist_type::Symbol=:surface,
                  maxit::Int=100, init_translation::Vector{Float64}=zeros(3),
                  iterfunc=nothing, lambda=nothing)

    ref_normals = dnormals(ref_cloud)

    inlier_sel = falses(length(cloud))
    T = init_translation

    for i = 1:maxit
        p = dpositions(cloud) .- T
        inds = find_closest_points(p, ref_cloud)
        v = p - dpositions(ref_cloud)[:,inds]

        # Compute residuals
        if dist_type == :point
            d = vec(sum(v.^2, 1))
            frmsd_lambda = 0.95
        elseif dist_type == :surface
            N = ref_normals[:,inds]
            d = vec(sum(v.*N, 1).^2)
            frmsd_lambda = 2.05 # point-to-surface gives statistics like 1D case
        else
            error("Unknown residual distance measure: $dist_type ")
        end
        if lambda !== nothing
            frmsd_lambda = lambda # override
        end

        # Detect inliers
        inlier_inds, f = frmsd_inliers(d, lambda=frmsd_lambda)

        iterfunc === nothing || iterfunc(i, T, inds, inlier_inds)


        # Optimal translation given match residuals.
        #
        # This problem has various names in various fields; on wikipedia it's
        # being called "Procrustes analysis":
        # http://en.wikipedia.org/wiki/Procrustes_analysis
        #
        if dist_type == :point
            # For the case of a simple 3D translation using total
            # point-to-point distance as the cost function, the least squares
            # solution for the translation is the average of the match
            # displacements.
            dT = squeeze(mean(v[:,inlier_inds], 2), 2)
        elseif dist_type == :surface
            # For the point-to-plane cost, the least squares solution is very
            # slightly more complex; here is the solution written in terms of
            # the normal equations:
            vi = v[:,inlier_inds]
            ni = N[:,inlier_inds]

            dT = squeeze(pinv(ni*ni') * sum((ni .* sum(ni.*vi, 1)), 2), 2)
        elseif dist_type == :point_3d
            # For the case of a full 3D rotation and translation, the translation
            # part can be estimated first, followed by solving the rotation using
            # SVD (the "Orthogonal Procrustes" problem).  Actually there's various
            # other ways to do this as well - see
            #
            # Ref: Eggert et al., "Estimating 3-D rigid body transformations: a
            # comparison of four major algorithms", Machine vision and
            # applications, p272 (1997).
            #
            # TODO!
        else
            error("Unknown residual distance measure: $dist_type")
        end

        T += dT
        inlier_sel[:] = false
        inlier_sel[inlier_inds] = true

        if norm(dT) < abstol
            break
        end
    end

    return T, inlier_sel
end

end
