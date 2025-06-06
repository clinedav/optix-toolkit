
// Ray with perpendicular axes
struct UVRay
{
    float3 O, D, U, V;
    float OdotU, OdotV;
};

static __forceinline__ __device__ float closestApproach( UVRay ray, const SphereParticle p )
{
    const float3 pC = float3{(float)p.C.x, (float)p.C.y, (float)p.C.z};
    const float invr2 = (float)p.C.w;

    float a = dot(pC, ray.U) - ray.OdotU;
    float b = dot(pC, ray.V) - ray.OdotV;
    return (a*a + b*b) * invr2;
}

static __forceinline__ __device__ float closestApproach( UVRay ray, const HalfSphereParticle p )
{
    const float3 pC = float3{(float)p.C.x, (float)p.C.y, (float)p.C.z};
    const float invr2 = (float)p.C.w;

    float a = dot(pC, ray.U) - ray.OdotU;
    float b = dot(pC, ray.V) - ray.OdotV;
    return (a*a + b*b) * invr2;
}


static __forceinline__ __device__ 
bool intersectHalfSphereParticle( Ray r, HalfSphereParticle p, Isect& isect )
{
    const float3 pC = float3{(float)p.C.x, (float)p.C.y, (float)p.C.z};
    const float invr2 = float(p.C.w);

    float a = dot(pC, r.U) - r.ROdotRU;
    float b = dot(pC, r.V) - r.ROdotRV;
    float d2 = (a*a + b*b) * invr2;

    if( d2 < 1.0f )
    {
        float tca = dot(pC - r.O, r.D);
        float alpha = (1.0f - d2) * (float)p.color.w;
        float3 color = float3{(float)p.color.x, (float)p.color.y, (float)p.color.z};
        isect.tval = tca;
        isect.color = uchar4{ (uchar)(color.x*255), (uchar)(color.y*255), (uchar)(color.z*255), (uchar)(alpha*255) };
        return true;
    }
    return false;
}