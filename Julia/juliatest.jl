using PhaseSpaceTools
using Plots

N=100
alpha = 1           # coherent amplitude
s = Coherent(alpha)     # define state |α⟩
a,b = husimiQ(s,N)

# scatter(real(a)+imag(a),(imag(a)-real(a)))
 
scatter(real(a),imag(a),xlabel="a",ylabel="a†",title="Sample for coherent State with α=1 \nin Husimi Q-Representation\nN=100")
savefig("husimiSample.pdf")