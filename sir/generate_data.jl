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
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "UK_data.csv"), data_UK)

vars = matread(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Rise_Korea_Track.mat"))
Infected = vars["Korea_Infected_All"]
Recovered = vars["Korea_Recovered_All"]
Dead= vars["Korea_Dead_All"]
Time = vars["Korea_Time"]
data_Korea = DataFrame(time = Time[:],
                       infected = Infected[:],
                       dead = Dead[:],
                       recovered = Recovered[:])
CSV.write(joinpath(@__DIR__, "..", "Intermediate", "SIR", "Korea_data.csv"), data_UK)
