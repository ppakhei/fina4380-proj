from mean_reverting_portfolio import mrp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    def plot_z():
        plt.figure(figsize=(10, 6))
        plt.plot((mrp_2000.mrp_value - mrp_2000.z_stat[0]) / mrp_2000.z_stat[1])
        plt.show()

    mrp_2000 = mrp(quantile=90, n=200)
    mrp_2000.update_portfolio(2000)
    plot_z()
