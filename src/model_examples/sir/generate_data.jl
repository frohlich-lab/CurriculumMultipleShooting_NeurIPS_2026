using OrdinaryDiffEq, CSV, DataFrames, Random, ModelingToolkit, MAT

vars = matread(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Rise_UK_Track.mat"))
Infected = vars["UK_Infected_All"]
Recovered = vars["UK_Recovered_All"]
Dead= vars["UK_Dead_All"]
Time = vars["UK_Time"]
data_UK = DataFrame(time = Time[:],
                    infected = Infected[:],
                    dead = Dead[:],
                    recovered = Recovered[:])
# In the original paper they do not have any validation data, naturally this is not
# ideal. Therefore we do a 80-20% split.
itrain = floor(length(Time)*0.8) |> Int64
data_UK_train = data_UK[1:itrain, :]
data_UK_validation = data_UK[(itrain+1):end, :]
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "UK_data.csv"), data_UK)
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "UK_data_train.csv"), data_UK_train)
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "UK_data_validation.csv"), data_UK_validation)

vars = matread(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Rise_Korea_Track.mat"))
Infected = vars["Korea_Infected_All"]
Recovered = vars["Korea_Recovered_All"]
Dead = vars["Korea_Dead_All"]
Time = vars["Korea_Time"]
data_Korea = DataFrame(time = Time[:],
                       infected = Infected[:],
                       dead = Dead[:],
                       recovered = Recovered[:])
# As above we make a 80-20% split of the data
itrain = floor(length(Time)*0.8) |> Int64
data_Korea_train = data_Korea[1:itrain, :]
data_Korea_validation = data_Korea[(itrain+1):end, :]
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Korea_data.csv"), data_Korea)
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Korea_data_train.csv"), data_Korea_train)
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Korea_data_validation.csv"), data_Korea_validation)
