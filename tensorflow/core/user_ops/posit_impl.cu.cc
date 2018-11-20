#include "posit_types.h"
#include <sys/param.h> 
#if 0
struct unpacked_t
{
    bool neg;
    POSIT_STYPE exp;
    POSIT_UTYPE frac;
};
#endif 
#define POW2(n) \
    (1 << (n))

#define FLOORDIV(a, b) \
    ((a) / (b) - ((a) % (b) < 0))

#define LMASK(bits, size) \
    ((bits) & (POSIT_MASK << (POSIT_WIDTH - (size))))

#define HIDDEN_BIT(frac) \
    (POSIT_MSB | ((frac) >> 1))

__device__ 
bool _util_is_neg(POSIT_UTYPE p)
{
    return (POSIT_STYPE)p < 0 && p != POSIT_INF;
}

__device__ 
int _util_ss()
{
    return 1;
}

__device__ 
int _util_rs(POSIT_UTYPE p, int nbits)
{
    int ss = _util_ss();
    int lz = __clz(p << ss);
    int lo = __clz(~p << ss);
    int rs = MAX(lz, lo) + 1;

    return MIN(rs, nbits - ss);
}

__device__ 
POSIT_UTYPE _util_neg(POSIT_UTYPE p, int nbits)
{
    // reverse all bits and add one
    return LMASK(-LMASK(p, nbits), nbits);
}

__device__ 
POSIT_UTYPE _pack_posit(struct unpacked_t up, int nbits, int es)
{
    POSIT_UTYPE p;
    POSIT_UTYPE regbits;
    POSIT_UTYPE expbits;

    // handle underflow and overflow.
    // in either case, exponent and fraction bits will disappear.
    int maxexp = POW2(es) * (nbits - 2);
    if (up.exp < -maxexp) {
        up.exp = -maxexp;
    } else if (up.exp > maxexp) {
        up.exp = maxexp;
    }

    int reg = FLOORDIV(up.exp, POW2(es));
    int ss = _util_ss();
    int rs = MAX(-reg + 1, reg + 2);

    // FIXME: round exponent up if needed
    if (ss + rs + es >= nbits && up.frac >= POSIT_MSB) {
        up.exp++;
        reg = FLOORDIV(up.exp, POW2(es));
        rs = MAX(-reg + 1, reg + 2);
    }

    POSIT_UTYPE exp = up.exp - POW2(es) * reg;

    if (reg < 0) {
        regbits = POSIT_MSB >> -reg;
    } else {
        regbits = LMASK(POSIT_MASK, reg + 1);
    }
    expbits = LMASK(exp << (POSIT_WIDTH - es), es);

    p = up.frac;
    p = expbits | (p >> es);
    p = regbits | (p >> rs);
    p = p >> ss;

    if (up.neg) {
        return _util_neg(p, nbits);
    } else {
        return LMASK(p, nbits);
    }
}

__device__ 
float _pack_float(struct unpacked_t up)
{
    int fexp = up.exp + 127;

    // left aligned
    uint32_t fexpbits;
    uint32_t ffracbits;

    if (fexp > 254) {
        // overflow, set maximum value
        fexpbits = 254 << 24;
        ffracbits = -1;
    } else if (fexp < 1) {
        // underflow, pack as denormal
        fexpbits = 0;
#if POSIT_WIDTH <= 32
        ffracbits = (uint32_t)(POSIT_MSB | (up.frac >> 1)) << (32 - POSIT_WIDTH);
#else
        ffracbits = (POSIT_MSB | (up.frac >> 1)) >> (POSIT_WIDTH - 32);
#endif
        ffracbits >>= -fexp;
    } else {
        fexpbits = (fexp & 0xFF) << 24;
#if POSIT_WIDTH <= 32
        ffracbits = (uint32_t)up.frac << (32 - POSIT_WIDTH);
#else
        ffracbits = up.frac >> (POSIT_WIDTH - 32);
#endif
    }

    union {
        float f;
        uint32_t u;
    } un;

    un.u = ffracbits;
    un.u = fexpbits | (un.u >> 8);
    un.u = (up.neg << 31) | (un.u >> 1);

    // don't underflow to zero
    if ((un.u << 1) == 0) {
        un.u++;
    }

    return un.f;
}

__device__ 
struct unpacked_t _unpack_posit(POSIT_UTYPE p, int nbits, int es)
{
    struct unpacked_t up;

    bool neg = _util_is_neg(p);
    if (neg) {
        p = _util_neg(p, nbits);
    }

    int ss = _util_ss();
    int rs = _util_rs(p, nbits);

    int lz = __clz(p << ss);
    int lo = __clz(~p << ss);

    int reg = (lz == 0 ? lo - 1 : -lz);
    POSIT_UTYPE exp = (p << (ss + rs)) >> (POSIT_WIDTH - es);

    up.neg = neg;
    up.exp = POW2(es) * reg + exp;
    up.frac = p << (ss + rs + es);

    return up;
}

__device__ 
struct unpacked_t _unpack_float(float f)
{
    struct unpacked_t up;
    int bias = 127;

    union {
        float f;
        uint32_t u;
    } un;

    un.f = f;

    up.neg = un.u >> 31;
    up.exp = ((un.u >> 23) & 0xFF) - bias;
#if POSIT_WIDTH <= 32
    up.frac = (un.u << 9) >> (32 - POSIT_WIDTH);
#else
    up.frac = (POSIT_UTYPE)un.u << (POSIT_WIDTH - 32 + 9);
#endif

    if (up.exp == -bias) {
        // normalize
        // FIXME: some precision is lost if frac was downcasted
        up.exp -= __clz(up.frac);
        up.frac <<= __clz(up.frac) + 1;
    }
    return up;
}
