      Program Xemiss
      parameter(nx=256)
      real xlum(nx,nx,nx)
c
      open(21,file='/storage/filament/data/256den18g+1024', form='FORMATTED')
      open(34,file='./out.dat',form='FORMATTED')
c
      read(21)zr
      read(21)time
      DO iz = 1, nx
        read(21)((xlum(ix,iy,iz),ix=1,nx),iy=1,nx)
      ENDDO
      print *, 'finished reading'
c: xlum = X-ray emissivity in units of 10^-30 ergs cm^-3 sec^-1
c
      flulim=1.e-5
      xlim = 1.e-5
c      rcube = 200.
      rcube = 100.
      zri = 51.59113
      zout = zr
      omega0 = 0.280
      h=0.7
      vol = (rcube/float(nx)*3.086/h)**3*1.0e-2 /(1.+zout)**3
      print *, 'vol', vol
c vol: volum in units of 10^{74} cm^3
c
      nl = 0
      DO  k = 1, nx
       print *, 'k, ', k
       DO  j = 1, nx
        DO  i = 1, nx
c
       if(xlum(i,j,k).gt.flumlim) then
       par = 1.0
c   Check if this cell is a local maximum
c       do 25  im= -5, 5
c       do 25  jm= -5, 5
c       do 25  km= -5, 5
       do 25  im= -10, 10
       do 25  jm= -10, 10
       do 25  km= -10, 10
       if((im.eq.0).and.(jm.eq.0).and.(km.eq.0)) go to 25
        in=i+im
        jn=j+jm
        kn=k+km
        if(in.le.0) in= in + nx
        if(jn.le.0) jn= jn + nx
        if(kn.le.0) kn= kn + nx
        if(in.ge.(nx+1)) in= in - nx
        if(jn.ge.(nx+1)) jn= jn - nx
        if(kn.ge.(nx+1)) kn= kn - nx
        if(xlum(i,j,k).gt.xlum(in,jn,kn)) then
         par=1.0
        else
         par=0.0
         go to 26
        endif
25     continue
26     continue

      if(par .eq.1.0) then
        xsum  = 0.0
c        do 255  im= -3, 3
c        do 255  jm= -3, 3
c        do 255  km= -3, 3
        do 255  im= -6, 6
        do 255  jm= -6, 6
        do 255  km= -6, 6
         dist = sqrt(float(im)**2 + float(jm)**2 + float(km)**2)
        if(dist.le. 5.12) then
        in=i+im
        jn=j+jm
        kn=k+km
        if(in.le.0) in= in + nx
        if(jn.le.0) jn= jn + nx
        if(kn.le.0) kn= kn + nx
        if(in.ge.(nx+1)) in= in - nx
        if(jn.ge.(nx+1)) jn= jn - nx
        if(kn.ge.(nx+1)) kn= kn - nx
        xsum = xsum + xlum (in,jn,kn)
        endif
255    continue
        xsum = xsum * vol
c       print *, 'i,j,k,xsum', i, j, k, xsum
c xsum : L_x in units of 10^{44} erg/s
        if (xsum.gt.1.0) then
          nl = nl +1
          write(34,500) nl, i,j,k, xsum
        endif
       endif
       endif
      ENDDO
      ENDDO
      ENDDO
500   format( 4i7, 1p1e15.6)
c
      stop
      end

