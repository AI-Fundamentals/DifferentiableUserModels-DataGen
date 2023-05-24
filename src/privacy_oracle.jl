export get_sigma

@doc raw"""
    get_sigma(epsilon, delta, num_repeats, subsampling_rate)
Computes the required privacy noise level using the privacy loss distribution.
Note: Internally, invokes the privacy_oracle.py Python script to perform the actual computation.
# Arguments
- `epsilon::Real`: Desired privacy parameter epsilon (0 <= `epsilon`).
- `delta::Real`: Desired privacy parameter delta (0 <= `delta` <= 1).
- `num_repeats::Integer`: Number of repeated queries to the data (e.g., SGD iterations).
- `subsampling_rate::Real`: The chance for any data point to be included in the query (0 <= `subsampling_rate` <= 1).
"""
function get_sigma(epsilon::Real, delta::Real, num_repeats::Integer, subsampling_rate::Real = 1.)
    cmd = `python3 src/privacy_oracle.py $epsilon $delta $num_repeats --subsampling_rate=$subsampling_rate`
    result = readchomp(cmd)
    return parse(Float64, result)
end
