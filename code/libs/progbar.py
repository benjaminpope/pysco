# Written by Guillaume, 2013

import os as _os
import sys as _sys

class progbar:
    """Progress bar
    *valmax* is the last value of the iteration. Default is 100.
    *barsize* is the size of the bar in the opened window. If empty, the bar will automatically fit the window
    *title* is the title."""
    def __init__(self, valmax=100, barsize=None, title='Be patient...'):
        self.reset(valmax,barsize,title)


    def reset(self, valmax=None, barsize=None, title=None):
        """Progress bar
        *valmax* is the last value of the iteration. Default is 100.
        *barsize* is the size of the bar in the opened window. If empty, the bar will automatically fit the window
        *title* is the title."""
        
        #gets terminal size
        env = _os.environ
        def ioctl_GWINSZ(fd):
            try:
                import fcntl, termi_os, struct, os
                cr = struct.unpack('hh', fcntl.ioctl(fd, termi_os.TIOCGWINSZ,
            '1234'))
            except:
                return
            return cr
        cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
        if not cr:
            try:
                fd = _os.open(_os.ctermid(), _os.O_RDONLY)
                cr = ioctl_GWINSZ(fd)
                _os.cl_ose(fd)
            except:
                pass
        if not cr:
            cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        #initialize
        self.win_width=int(cr[1])
        self.in_progress=False
        self.val=0
        valmax_std=100
        barsize_max=self.win_width-8
        if not hasattr(self,'valmax'): self.valmax=valmax_std
        if valmax is not None: self.valmax=valmax+(valmax==0)*valmax_std
        if not hasattr(self,'barsize'): self.barsize=barsize_max-8
        if barsize is not None: self.barsize=min(barsize,barsize_max)
        if not hasattr(self,'title'): self.title=title
        if title is not None: self.title=title

    
    def update(self, val='add'):
        """Calling .update() adds 1 to the progress of the bar.
        Calling .update(i) puts the progress of the bar to i value."""
        # update value of bar
        self.val=(isinstance(val,str)==False)*min(val,self.valmax)+(isinstance(val,str))*(1+self.val)
        
        # process
        perc  = round((float(self.val) / float(self.valmax)) * 100)
        bar   = int(perc *float(self.barsize) / 100)
  
        # render
        if self.in_progress is False:
            print self.title
            self.in_progress=True
        spacing=(self.win_width-self.barsize-8)/2.
        out = '\r%s[%s%s] %3d %%%s' % (' ' * int(spacing), '=' * bar, ' ' * (self.barsize - bar), perc, ' ' * int(spacing+0.5))
        _sys.stdout.write(out)
        _sys.stdout.flush()
