from visdom import Visdom

class VisdomPlotter():
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot_img(self, np_arr, win=None, caption=None):
        return self.viz.image(
            np_arr,
            win=win,
            opts = dict(caption=caption, width=250, height=250)
        )

    def plot_line(self, X, Y, title, win = None, update=False,xlabel=None,ylabel=None,legend=None):
        if update:
            return self.viz.line(
                Y=Y,
                X=X,
                win=win,
                opts=dict(showlegend=True,title=title, markers=False,
                xlabel=xlabel, ylabel=ylabel, legend=legend, marginleft=30,
                width=250, height=250),
                update='append'
            )
        else:
            return self.viz.line(
                Y=Y,
                X=X,
                win=win,
                opts=dict(showlegend=True,title=title, markers=False,xlabel=xlabel, ylabel=ylabel, legend=legend, width=250, height=250)
            )

        # {'legend': {'x':0, 'y':0}}}
            
    # def plot(self, var_name, split_name, title_name, x, y):
    #     if var_name not in self.plots:
    #         self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
    #             legend=[split_name],
    #             title=title_name,
    #             xlabel='Epochs',
    #             ylabel=var_name
    #         ))
    #     else:
    #         self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')