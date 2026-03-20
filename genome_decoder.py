"""
Genome Decoder: Binary bitstring -> real-valued neural network parameters.

Uses Gray code decoding so that single bit flips cause small parameter changes,
preserving the neutrality structure of the existing GA (Alife_Assignment).

Gray code ensures adjacent binary strings map to adjacent real values,
meaning neutral mutations (bit flips that don't change fitness) are more
likely to occur between neighbouring genotypes.
"""
import numpy as np


def gray_to_binary(gray_bits):
    """Convert Gray code bit array to standard binary integer."""
    binary = np.zeros_like(gray_bits)
    binary[0] = gray_bits[0]
    for i in range(1, len(gray_bits)):
        binary[i] = binary[i - 1] ^ gray_bits[i]
    return binary


def bits_to_float(bits, lo, hi):
    """
    Decode a block of bits to a float in [lo, hi] via Gray code.

    Args:
        bits: 1D binary numpy array
        lo: lower bound of real range
        hi: upper bound of real range

    Returns:
        float in [lo, hi]
    """
    n = len(bits)
    binary = gray_to_binary(bits)
    # Convert binary array to integer
    integer = int(''.join(str(int(b)) for b in binary), 2)
    max_val = (2 ** n) - 1
    return lo + (integer / max_val) * (hi - lo)


class GenomeDecoder:
    """
    Decodes a flat binary genome into neural network weights.

    The genome is partitioned into fixed-size blocks, one per parameter.
    Each block is Gray-decoded to a float in a specified range.
    Unused/padding bits at the end of the genome are neutral by construction.

    Args:
        param_specs: list of (name, count, lo, hi, bits_per_param)
            - name: label (e.g. 'W', 'bias')
            - count: number of parameters in this group
            - lo/hi: value range
            - bits_per_param: number of bits used to encode each parameter
    """

    def __init__(self, param_specs):
        self.param_specs = param_specs
        self.N = sum(spec[1] * spec[4] for spec in param_specs)

    @property
    def N_params(self):
        """Total number of real-valued parameters (for real-valued encoding)."""
        return sum(spec[1] for spec in self.param_specs)

    def decode_realvalued(self, genome):
        """
        Decode a normalised [0, 1] float genome to a dict of parameter arrays.
        Each gene value v in [0, 1] maps linearly to the parameter's [lo, hi] range.
        Used by GENOME_ENCODING='realvalued' — no bit manipulation needed.
        """
        params = {}
        offset = 0
        for (name, count, lo, hi, _) in self.param_specs:
            params[name] = lo + genome[offset:offset + count] * (hi - lo)
            offset += count
        return params

    def decode(self, genome):
        """
        Decode binary genome to dict of named parameter arrays.

        Args:
            genome: binary numpy array of length >= self.N

        Returns:
            dict mapping name -> numpy array of floats
        """
        params = {}
        offset = 0
        for (name, count, lo, hi, bpp) in self.param_specs:
            values = []
            for _ in range(count):
                block = genome[offset: offset + bpp]
                values.append(bits_to_float(block, lo, hi))
                offset += bpp
            params[name] = np.array(values)
        # Remaining bits (if any) are neutral padding — intentionally ignored
        return params


def build_decoder_for_controller(controller_type, obs_dim, act_dim,
                                 hidden_size, bits_per_weight=16):
    """
    Build a GenomeDecoder for either 'feedforward' or 'ctrnn'.

    Feedforward params:
        W1 (obs_dim * hidden_size), b1 (hidden_size),
        W2 (hidden_size * act_dim),  b2 (act_dim)

    CTRNN params:
        W  (hidden_size * hidden_size),  W_in (obs_dim * hidden_size),
        W_out (hidden_size * act_dim),   bias (hidden_size),
        tau (hidden_size)

    Weight range: [-3, 3]  Bias range: [-2, 2]  Tau range: [0.1, 2.0]

    Args:
        controller_type: 'feedforward' or 'ctrnn'
        obs_dim: input dimension
        act_dim: output (action) dimension
        hidden_size: number of hidden/recurrent units
        bits_per_weight: bits allocated per parameter (default 16)

    Returns:
        GenomeDecoder instance
    """
    W_LO, W_HI = -3.0, 3.0
    B_LO, B_HI = -2.0, 2.0
    T_LO, T_HI = 0.1, 2.0
    bpw = bits_per_weight

    if controller_type == 'feedforward':
        specs = [
            ('W1',   obs_dim * hidden_size,    W_LO, W_HI, bpw),
            ('b1',   hidden_size,              B_LO, B_HI, bpw),
            ('W2',   hidden_size * act_dim,    W_LO, W_HI, bpw),
            ('b2',   act_dim,                  B_LO, B_HI, bpw),
        ]
    elif controller_type == 'ctrnn':
        specs = [
            ('W',     hidden_size * hidden_size, W_LO, W_HI, bpw),
            ('W_in',  obs_dim * hidden_size,     W_LO, W_HI, bpw),
            ('W_out', hidden_size * act_dim,     W_LO, W_HI, bpw),
            ('bias',  hidden_size,               B_LO, B_HI, bpw),
            ('tau',   hidden_size,               T_LO, T_HI, bpw),
        ]
    else:
        raise ValueError(f"Unknown controller_type '{controller_type}'. "
                         "Choose 'feedforward' or 'ctrnn'.")

    return GenomeDecoder(specs)
