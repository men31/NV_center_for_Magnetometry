from statistics import variance
import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, number):
        self.number = number
        self.X, self.O, self.V = 0, np.inf, 0
        self.PBEST_X, self.PBEST_O = 0, np.inf


class Commander:
    def __init__(self, noAgents, lowbound, upbound, InitialPoint):
        self.noAgents = noAgents
        self.lowbound, self.upbound = lowbound, upbound
        self.numVar = len(self.upbound)
        self.InitialPoint = InitialPoint

        self.Agents = {num: Agent(num) for num in range(noAgents)}

        self.GBEST_X, self.GBEST_O = 0, np.inf

        # Initialize for every agents
        if self.InitialPoint is None:
            for num in range(noAgents):
                self.Agents[num].X = (self.upbound - self.lowbound) * \
                    np.random.rand(self.numVar) + self.lowbound
                self.Agents[num].V = np.zeros((self.numVar))
                self.Agents[num].PBEST_X = np.zeros((self.numVar))
                self.Agents[num].PBEST_O = np.inf

                self.GBEST_X = np.zeros((self.numVar))
                self.GBEST_O = np.inf
        else:
            for num in range(noAgents):
                self.Agents[num].X = np.random.choice(
                    np.linspace(-0.5, 0.5, 100), self.numVar) + self.InitialPoint
                self.Agents[num].V = np.zeros((self.numVar))
                self.Agents[num].PBEST_X = np.zeros((self.numVar))
                self.Agents[num].PBEST_O = np.inf

                self.GBEST_X = np.zeros((self.numVar))
                self.GBEST_O = np.inf


class PSO:
    def __init__(self, Objfunc, lowbound, upbound, maxIter, InitialPoint=None):

        self.maxIter = maxIter
        # To flat the boundary conditions
        self.lowbound, self.upbound = lowbound, upbound
        self.numVar = len(self.upbound)
        self.ObjFunc = Objfunc

        # Define the parameter for optimizing
        self.noAgents = 50
        self.Commander = Commander(
            self.noAgents,  self.lowbound, self.upbound, InitialPoint)
        self.wMax = 0.9
        self.wMin = 0.2
        self.c1, self.c2 = 2, 3
        self.vMax = (self.upbound - self.lowbound) * 0.2
        self.vMin = - self.vMax

        # Ouput parameters
        self.have_calculated = False
        self.history = np.array([])

    def Calculate(self):
        for i in range(self.maxIter):
            # Calculate the objective value
            for num in range(self.noAgents):
                currentX = self.Commander.Agents[num].X
                self.Commander.Agents[num].O = self.ObjFunc(currentX)

                # Update the Agent properties
                if self.Commander.Agents[num].O < self.Commander.Agents[num].PBEST_O:
                    self.Commander.Agents[num].PBEST_X = currentX
                    self.Commander.Agents[num].PBEST_O = self.Commander.Agents[num].O

                # Update the Global properties
                if self.Commander.Agents[num].O < self.Commander.GBEST_O:
                    self.Commander.GBEST_X = currentX
                    self.Commander.GBEST_O = self.Commander.Agents[num].O

            w = self.wMax - i * (self.wMax - self.wMin) / self.maxIter

            # Update the Agent position and velocity
            for num in range(self.noAgents):

                # Update velocity
                self.Commander.Agents[num].V = w * self.Commander.Agents[num].V + self.c1 * \
                    np.random.rand(self.numVar) * \
                    (self.Commander.Agents[num].PBEST_X - self.Commander.Agents[num].X) + \
                    self.c2 * np.random.rand(self.numVar) * (
                        self.Commander.GBEST_X - self.Commander.Agents[num].X)

                # Check the velocity
                idx_vmin = np.where(self.Commander.Agents[num].V < self.vMin)
                idx_vmax = np.where(self.Commander.Agents[num].V > self.vMax)

                self.Commander.Agents[num].V[idx_vmin] = self.vMin[idx_vmin]
                self.Commander.Agents[num].V[idx_vmax] = self.vMax[idx_vmax]

                # Update the Agent postion
                self.Commander.Agents[num].X = self.Commander.Agents[num].X + \
                    self.Commander.Agents[num].V

                # Check the postion
                idx_xmin = np.where(
                    self.Commander.Agents[num].X < self.lowbound)
                idx_xmax = np.where(
                    self.Commander.Agents[num].X > self.upbound)

                self.Commander.Agents[num].V[idx_xmin] = self.lowbound[idx_xmin]
                self.Commander.Agents[num].V[idx_xmax] = self.upbound[idx_xmax]

            self.history = np.append(self.history, self.Commander.GBEST_O)

        self.have_calculated = True
        return self.Commander.GBEST_X, self.Commander.GBEST_O

    def Plot_History(self):
        if not self.have_calculated:
            _, _ = self.Calculate()

        plt.plot(np.arange(self.maxIter), self.history,
                 '.-', label='loss value')
        plt.title('History of loss value')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    def ObjectiveFunc(X):
        return -4*X[0] / (X[0]**2 + X[1]**2 + 1)
    lb = np.array([-5, -8])
    ub = np.array([5, 3])
    maxIter = 500
    Particle_SO = PSO(ObjectiveFunc, lb, ub, maxIter)
    GBEST_X, GBEST_O = Particle_SO.Calculate()
    print('Best position : ', GBEST_X)
    print('Objective value : ', GBEST_O)
    Particle_SO.Plot_History()
