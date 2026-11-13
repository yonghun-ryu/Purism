`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
// VX_ocu_converters.sv — INT→INT, normalizer, INT→FP, FP→FP sub-modules
// ---------------------------------------------------------------------------



// ===========================================================================
// VX_ocu_int2int
// INT → INT converter (zero-point subtraction + sign extension)
// critical path: n-bit subtraction (the simplest path)
// ---------------------------------------------------------------------------
module VX_ocu_int2int
    import VX_ocu_pkg::*;
#(
    /* verilator lint_off UNUSEDPARAM */
    parameter int MAX_SRC_BW = 8,   // unused: int2int uses SRC_DATA_WIDTH pkg constant
    /* verilator lint_on UNUSEDPARAM */
    parameter int MAX_DST_BW = 16
)(
    input  logic [SRC_DATA_WIDTH-1:0] src_data,
    input  logic [ZP_WIDTH-1:0]       zero_point,
    /* verilator lint_off UNUSEDSIGNAL */
    input  fmt_ctrl_t                  ctrl,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [DST_DATA_WIDTH-1:0]  result
);
    // -----------------------------------------------------------------------
    // internal signals
    // -----------------------------------------------------------------------

    // zero-point subtraction result (up to ZP_WIDTH+1 bits, no overflow)
    logic [ZP_WIDTH:0] zp_sub;

    // sign extension result
    logic [MAX_DST_BW-1:0] sign_ext;

    // -----------------------------------------------------------------------
    // Stage 1: Mask + Sign-Extend + Zero-Point Subtraction
    //   src and ZP are both masked to src_bw bits.
    //   signed formats: both are sign-extended from src_bw to ZP_WIDTH, then subtracted.
    //   unsigned formats: masking only, no sign extension.
    //   8-bit mask corner case: src_bw=8 → (1<<8)=0 (overflow), 0-1=0xFF ✓
    // -----------------------------------------------------------------------
    logic [ZP_WIDTH-1:0] src_masked;
    logic [ZP_WIDTH-1:0] bw_mask;

    always_comb begin
        logic [ZP_WIDTH-1:0] zp_masked;
        logic [ZP_WIDTH-1:0] src_ext, zp_ext;
        logic                src_sign_bit, zp_sign_bit;

        bw_mask      = (ZP_WIDTH'(1) << ctrl.src_bw) - 1;
        src_masked   = src_data[ZP_WIDTH-1:0] & bw_mask;
        zp_masked    = zero_point & bw_mask;

        src_sign_bit = ctrl.src_signed ? src_masked[ctrl.src_bw - 1] : 1'b0;
        zp_sign_bit  = ctrl.src_signed ? zp_masked[ctrl.src_bw - 1]  : 1'b0;
        src_ext      = src_masked | ({ZP_WIDTH{src_sign_bit}} & ~bw_mask);
        zp_ext       = zp_masked  | ({ZP_WIDTH{zp_sign_bit}}  & ~bw_mask);

        // 9-bit sign-extended subtraction: src_sign_bit/zp_sign_bit already encode
        // signedness, so {src_sign_bit, src_ext} correctly sign- or zero-extends.
        // For src_bw=8 (I8/U8), ~bw_mask=0 and the old 8-bit result MSB was wrong.
        zp_sub       = {src_sign_bit, src_ext} - {zp_sign_bit, zp_ext};
    end

    // -----------------------------------------------------------------------
    // Stage 2: Sign Extension to MAX_DST_BW
    //   Stage 1 produces a 9-bit sign-extended difference,
    //   so the sign is extended from zp_sub[ZP_WIDTH] (the borrow bit).
    // -----------------------------------------------------------------------
    always_comb begin
        logic        sign_bit;
        logic [MAX_DST_BW-1:0] upper_fill;

        sign_bit   = zp_sub[ZP_WIDTH];
        upper_fill = {MAX_DST_BW{sign_bit}} & ~((MAX_DST_BW'(1) << ZP_WIDTH) - 1);

        sign_ext = MAX_DST_BW'(zp_sub) | upper_fill;
        result   = sign_ext;
        // 8-bit destination: saturate to [-128, 127] (I8/U8 sources minus a zero
        // point can leave the INT8 range; 4-bit sources never do).
        if (ctrl.dst_bw <= 5'd8) begin
            if ($signed(zp_sub) > $signed(9'sd127)) begin
                result = MAX_DST_BW'(8'sd127);
            end else if ($signed(zp_sub) < -$signed(9'sd128)) begin
                result = {{(MAX_DST_BW-8){1'b1}}, 8'h80};
            end
        end
    end

endmodule


// ===========================================================================
// VX_ocu_normalizer
// Fused Leading-One Detect + Normalization (shared by int2fp and fp2fp)
//
// 3-level binary MUX tree: simultaneously produces exp_index (leading-1
// position) and norm_frac (bits below leading-1, MSB-aligned).
// Eliminates the separate priority-encoder → subtractor → barrel-shift chain.
//
// Critical path: OR(L1) → MUX(L1) → OR(L2) → MUX(L2) → MUX(L3)  ≈ 45ps
// ---------------------------------------------------------------------------
module VX_ocu_normalizer #(
    parameter int MAX_SRC_BW = 8   // fixed 8-bit; 3-level tree
)(
    input  logic [MAX_SRC_BW-1:0]   magnitude,  // already masked to src_bw bits
    output logic [2:0]               exp_index,  // leading-1 position (0-indexed from LSB)
    output logic [MAX_SRC_BW-2:0]   norm_frac   // bits below leading-1, MSB-aligned
);
    logic [MAX_SRC_BW-1:0] mid1, mid2;

    always_comb begin
        logic l1_sel, l2_sel, l3_sel;

        // Level 1: if upper half [7:4] has a 1, leading-1 is in bits 4..7 (no shift).
        // Otherwise shift left by 4 to bring lower half into upper position.
        l1_sel = |magnitude[7:4];
        mid1   = l1_sel ? magnitude : {magnitude[3:0], 4'b0};

        // Level 2: leading-1 is now in mid1[7:4].
        // If mid1[7:6] has a 1, leading-1 is in bits 6..7 (no shift).
        // Otherwise shift left by 2.
        l2_sel = |mid1[7:6];
        mid2   = l2_sel ? mid1 : {mid1[5:0], 2'b0};

        // Level 3: leading-1 is now in mid2[7:6].
        // If mid2[7]=1, leading-1 is at bit 7 (no shift). Otherwise shift left by 1.
        l3_sel = mid2[7];

        // exp_index: concatenation of the three level selectors encodes the
        // leading-1 position directly (no subtractor needed).
        exp_index = {l1_sel, l2_sel, l3_sel};

        // norm_frac: bits below the implicit leading-1 (always at MSB after normalization).
        // Direct assignment avoids a redundant 'normalized' intermediate whose MSB
        // would always be 1 (the implicit leading-1 that gets stripped).
        norm_frac = l3_sel ? mid2[MAX_SRC_BW-2:0] : {mid2[MAX_SRC_BW-3:0], 1'b0};
    end
endmodule


// ===========================================================================
// VX_ocu_int2fp
// INT → FP converter
// critical path (n=8): ZP_sub(240ps) → Norm-tree(45ps) → exp_add(150ps) → MUX(15ps) = 450ps
//   Fused normalizer removes priority-encoder + subtractor from the chain.
//   Tree depth: 3 levels (OR+MUX each) ≈ 45ps.
// ---------------------------------------------------------------------------
module VX_ocu_int2fp
    import VX_ocu_pkg::*;
#(
    parameter int MAX_SRC_BW = 8,
    /* verilator lint_off UNUSEDPARAM */
    parameter int MAX_DST_BW = 16  // unused: int2fp uses DST_DATA_WIDTH pkg constant
    /* verilator lint_on UNUSEDPARAM */
)(
    input  logic [SRC_DATA_WIDTH-1:0] src_data,
    input  logic [ZP_WIDTH-1:0]       zero_point,
    /* verilator lint_off UNUSEDSIGNAL */
    input  fmt_ctrl_t                  ctrl,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [DST_DATA_WIDTH-1:0]  result
);

    // -----------------------------------------------------------------------
    // Stage A: ZP Subtraction + Sign/Magnitude (carry-in trick, 1 adder)
    //   src and ZP are both masked to src_bw bits (same precision assumed).
    //   signed formats: both sign-extended, then subtracted → sign = MSB of result.
    //   unsigned formats: zero-extended, then subtracted → sign = MSB of result.
    //   magnitude = |zp_corrected| via XOR + carry-in (1 adder)
    // -----------------------------------------------------------------------
    logic [MAX_SRC_BW-1:0] zp_corrected;
    logic                   sign;
    logic [MAX_SRC_BW-1:0] magnitude;
    logic                   zero_flag;

    always_comb begin
        logic [MAX_SRC_BW-1:0] bw_mask;
        logic [MAX_SRC_BW-1:0] src_masked, zp_masked;
        logic                   src_sign_bit, zp_sign_bit;
        logic [MAX_SRC_BW-1:0] src_ext, zp_ext;
        logic [MAX_SRC_BW:0]   sub9;

        bw_mask      = (MAX_SRC_BW'(1) << ctrl.src_bw) - 1;
        src_masked   = src_data[MAX_SRC_BW-1:0] & bw_mask;
        zp_masked    = MAX_SRC_BW'(zero_point)  & bw_mask;

        src_sign_bit = ctrl.src_signed ? src_masked[ctrl.src_bw - 1] : 1'b0;
        zp_sign_bit  = ctrl.src_signed ? zp_masked[ctrl.src_bw - 1]  : 1'b0;
        src_ext      = src_masked | ({MAX_SRC_BW{src_sign_bit}} & ~bw_mask);
        zp_ext       = zp_masked  | ({MAX_SRC_BW{zp_sign_bit}}  & ~bw_mask);

        // 9-bit sign-extended subtraction: for src_bw=8 (I8/U8), ~bw_mask=0 so
        // the old 8-bit result MSB (bit7) could not distinguish sign from magnitude.
        // Borrow bit sub9[MAX_SRC_BW] is the true sign for all src_bw 4..8.
        sub9 = {src_sign_bit, src_ext} - {zp_sign_bit, zp_ext};
        zp_corrected = sub9[MAX_SRC_BW-1:0];

        // sign: borrow bit from 9-bit subtraction
        sign = sub9[MAX_SRC_BW];

        // magnitude: carry-in trick (XOR mask + sign as carry-in = 1 adder)
        begin
            logic [MAX_SRC_BW-1:0] xor_mask;
            logic [MAX_SRC_BW-1:0] flipped;
            xor_mask  = {MAX_SRC_BW{sign}};
            flipped   = zp_corrected ^ xor_mask;
            magnitude = flipped + MAX_SRC_BW'(sign);
        end

        zero_flag = (magnitude == '0);
    end

    // -----------------------------------------------------------------------
    // Stage B: Fused LOD/Normalize via VX_ocu_normalizer
    //   exp_index = leading-1 position in magnitude (0-indexed from LSB)
    //   norm_frac = bits below leading-1, MSB-aligned in MAX_SRC_BW-1 bits
    // -----------------------------------------------------------------------
    logic [2:0]             exp_index;
    logic [MAX_SRC_BW-2:0]  norm_frac;

    VX_ocu_normalizer #(.MAX_SRC_BW(MAX_SRC_BW)) u_norm (
        .magnitude (magnitude),
        .exp_index (exp_index),
        .norm_frac (norm_frac)
    );

    // -----------------------------------------------------------------------
    // Stage C: Exponent Assembly + Mantissa Packing
    //   exp_out  = exp_index + dst_bias  (no subtractor; exp_index directly from normalizer)
    //   norm_frac is 7-bit MSB-aligned with implicit-1 stripped — zero-pad to 10 bits,
    //   right-shift to LSB-align in dst_ms bits.
    // -----------------------------------------------------------------------
    logic [4:0]  exp_out;
    logic [9:0]  mant_lsb;

    always_comb begin
        logic [DST_DATA_WIDTH-1:0] fp_word;
        logic [4:0]                exp_masked;
        logic [9:0]                mant_masked;
        logic [4:0]                pack_shift;

        fp_word     = '0;
        exp_masked  = '0;
        mant_masked = '0;
        pack_shift  = '0;

        // 5'(): no overflow (max = exp_index(7) + dst_bias(15) = 22 < 32)
        exp_out  = 5'({3'b0, exp_index} + {1'b0, ctrl.dst_bias});
        mant_lsb = {norm_frac, 3'b0} >> (10 - ctrl.dst_ms);

        if (zero_flag) begin
            result = '0;
        end else begin
            pack_shift  = 5'(DST_DATA_WIDTH - 1) - {2'b0, ctrl.dst_es} - {1'b0, ctrl.dst_ms};
            exp_masked  = exp_out  & ((5'b1  << ctrl.dst_es) - 1);
            mant_masked = mant_lsb & ((10'b1 << ctrl.dst_ms) - 1);

            fp_word = DST_DATA_WIDTH'(sign);
            fp_word = (fp_word << ctrl.dst_es) | DST_DATA_WIDTH'(exp_masked);
            fp_word = (fp_word << ctrl.dst_ms) | DST_DATA_WIDTH'(mant_masked);
            result  = fp_word << pack_shift;
        end
    end

endmodule


// ===========================================================================
// VX_ocu_fp2fp
// FP → FP converter (exponent rebias + mantissa zero-pad + subnormal handling)
// critical path: field split(wire) → rebias adder(150ps) → MUX(15ps) = 165ps
//                subnormal path: + LOD(30ps) + shift(30ps) = 225ps
// ---------------------------------------------------------------------------
module VX_ocu_fp2fp
    import VX_ocu_pkg::*;
#(
    parameter int MAX_SRC_BW = 8,
    /* verilator lint_off UNUSEDPARAM */
    parameter int MAX_DST_BW = 16  // unused: fp2fp uses DST_DATA_WIDTH pkg constant
    /* verilator lint_on UNUSEDPARAM */
)(
    input  logic [SRC_DATA_WIDTH-1:0] src_data,
    /* verilator lint_off UNUSEDSIGNAL */
    input  fmt_ctrl_t                  ctrl,
    /* verilator lint_on UNUSEDSIGNAL */
    output logic [DST_DATA_WIDTH-1:0]  result
);

    // -----------------------------------------------------------------------
    // Stage 1: Field Split (wire, 0 latency)
    //   src_data = {sign[1], exponent[src_es], mantissa[src_ms]}
    //   field boundaries are set at run time by ctrl.src_es, ctrl.src_ms
    // -----------------------------------------------------------------------
    logic                   fp_sign;
    logic [4:0]             fp_exp_raw;   // up to 5 bits (FP16)
    logic [6:0]             fp_mant_raw;  // up to 7 bits (extreme cases such as FP8 E1M7)
    logic                   is_subnormal; // exponent == 0
    logic                   is_special;   // exponent == all-ones (NaN / Inf)

    always_comb begin
        fp_sign = src_data[ctrl.src_bw - 1];

        // explicit 32-bit cast: shift+mask the 8-bit src_data in a 32-bit context
        fp_exp_raw  = 5'((32'(src_data) >> ctrl.src_ms) & ((32'd1 << ctrl.src_es) - 1));
        fp_mant_raw = 7'(32'(src_data) & ((32'd1 << ctrl.src_ms) - 1));

        is_subnormal = (fp_exp_raw == 5'b0);
        // OCP FP8 E4M3 (src_es==4): NaN only when BOTH exp==all-ones AND mant==all-ones.
        // Values with exp==0xF and mant!=all-ones (e.g. 256..448) are finite normals.
        // OCP FP4 E2M1 (src_es==2): no Inf/NaN, exp==all-ones is ±4, ±6.
        // E5M2 and all other formats follow IEEE: exp==all-ones → always special.
        is_special   = (fp_exp_raw == 5'((32'd1 << ctrl.src_es) - 1))
                     && (ctrl.src_es != 3'd2)
                     && ((ctrl.src_es != 3'd4)
                         || (fp_mant_raw == 7'((32'd1 << ctrl.src_ms) - 1)));
    end

    // -----------------------------------------------------------------------
    // Stage 2a: Normal path — exponent rebias + mantissa zero-pad
    // -----------------------------------------------------------------------
    logic [4:0] exp_rebiased;
    logic [9:0] mant_padded;

    always_comb begin
        // exponent rebias: exp_dst = exp_src + bias_delta
        exp_rebiased = 5'($signed({1'b0, fp_exp_raw}) + $signed(ctrl.bias_delta));

        // mantissa: keep the upper src_ms bits, zero-pad the lower (dst_ms - src_ms) bits
        mant_padded  = 10'(fp_mant_raw) << (ctrl.dst_ms - ctrl.src_ms);
    end

    // -----------------------------------------------------------------------
    // Stage 2b: Subnormal normalization
    //   src subnormal (exp=0): value = (-1)^sign × 2^(1-src_bias) × 0.fp_mant_raw
    //
    //   Reuses VX_ocu_normalizer on fp_mant_raw (7-bit, zero-extended to 8).
    //   lzc       = (src_ms - 1) - subnorm_exp_idx
    //   exp_dst   = bias_delta - lzc
    //
    //   If exp_dst ≤ 0 (src subnormal below dst representable range), result
    //   is clamped to zero. Full subnormal-in-dst support is a future TODO.
    // -----------------------------------------------------------------------
    logic [2:0]              subnorm_exp_idx;
    logic [MAX_SRC_BW-2:0]  subnorm_norm_frac;
    logic [DST_DATA_WIDTH-1:0] subnormal_result;

    VX_ocu_normalizer #(.MAX_SRC_BW(MAX_SRC_BW)) u_subnorm (
        .magnitude ({1'b0, fp_mant_raw}),  // 7-bit mant zero-extended to 8 bits
        .exp_index (subnorm_exp_idx),
        .norm_frac (subnorm_norm_frac)
    );

    always_comb begin
        logic [3:0]        subnorm_lzc;
        logic signed [6:0] exp_s;
        logic [4:0]        exp_subnorm;
        logic [9:0]        mant_subnorm_lsb;
        logic [DST_DATA_WIDTH-1:0] fp_word;
        logic [4:0]        exp_masked;
        logic [9:0]        mant_masked;
        logic [4:0]        pack_shift;

        subnorm_lzc      = '0;
        exp_s            = '0;
        exp_subnorm      = '0;
        mant_subnorm_lsb = '0;
        fp_word          = '0;
        exp_masked       = '0;
        mant_masked      = '0;
        pack_shift       = '0;
        subnormal_result = '0;

        if (fp_mant_raw == '0) begin
            // exp=0, mant=0 → signed zero
            subnormal_result = DST_DATA_WIDTH'(fp_sign) << (DST_DATA_WIDTH - 1);
        end else begin
            subnorm_lzc = 4'(ctrl.src_ms) - 4'd1 - {1'b0, subnorm_exp_idx};
            exp_s       = {{1{ctrl.bias_delta[5]}}, ctrl.bias_delta}
                        - {3'b0, subnorm_lzc};
            exp_subnorm = (exp_s <= 0) ? 5'b0 : 5'(exp_s[4:0]);

            mant_subnorm_lsb = {subnorm_norm_frac, 3'b0} >> (10 - ctrl.dst_ms);
            if (exp_s <= 0) begin
                // below the destination normal range (e.g. E5M2 -> FP16, equal bias):
                // emit a destination subnormal, significand 1.f shifted right by 1-exp_s
                mant_subnorm_lsb = 10'({1'b1, mant_subnorm_lsb} >> (7'sd1 - exp_s));
            end

            pack_shift  = 5'(DST_DATA_WIDTH - 1) - {2'b0, ctrl.dst_es} - {1'b0, ctrl.dst_ms};
            exp_masked  = exp_subnorm      & ((5'b1  << ctrl.dst_es) - 1);
            mant_masked = mant_subnorm_lsb & ((10'b1 << ctrl.dst_ms) - 1);

            fp_word = DST_DATA_WIDTH'(fp_sign);
            fp_word = (fp_word << ctrl.dst_es) | DST_DATA_WIDTH'(exp_masked);
            fp_word = (fp_word << ctrl.dst_ms) | DST_DATA_WIDTH'(mant_masked);
            subnormal_result = fp_word << pack_shift;
        end
    end

    // -----------------------------------------------------------------------
    // Stage 3: Output MUX (normal / subnormal / special / zero)
    //   no variable-width part-select → MSB-first shift-build
    //   mant_lsb: mant_padded (MSB-aligned 10-bit) → right-shift to LSB-align in dst_ms bits
    // -----------------------------------------------------------------------
    always_comb begin
        logic [DST_DATA_WIDTH-1:0] fp_word;
        logic [4:0]                exp_masked;
        logic [9:0]                mant_lsb;
        logic [4:0]                pack_shift;

        fp_word    = '0;
        exp_masked = '0;
        pack_shift = '0;

        mant_lsb   = mant_padded >> (10 - ctrl.dst_ms);
        pack_shift = 5'(DST_DATA_WIDTH - 1) - {2'b0, ctrl.dst_es} - {1'b0, ctrl.dst_ms};
        result     = '0;

        if (is_subnormal) begin
            result = subnormal_result;

        end else if (is_special) begin
            exp_masked = (5'b1 << ctrl.dst_es) - 1;
            fp_word = DST_DATA_WIDTH'(fp_sign);
            fp_word = (fp_word << ctrl.dst_es) | DST_DATA_WIDTH'(exp_masked);
            fp_word = (fp_word << ctrl.dst_ms) | DST_DATA_WIDTH'(mant_lsb);
            result  = fp_word << pack_shift;

        end else begin
            exp_masked = 5'(exp_rebiased) & ((5'b1 << ctrl.dst_es) - 1);
            fp_word = DST_DATA_WIDTH'(fp_sign);
            fp_word = (fp_word << ctrl.dst_es) | DST_DATA_WIDTH'(exp_masked);
            fp_word = (fp_word << ctrl.dst_ms) | DST_DATA_WIDTH'(mant_lsb);
            result  = fp_word << pack_shift;
        end
    end

endmodule
