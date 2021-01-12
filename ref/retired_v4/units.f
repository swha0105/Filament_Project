      subroutine units
c
c-----------------------------------------------------------------------
c     this subroutine changes the units in the routines
c     to the real physical units
c-----------------------------------------------------------------------
c
      implicit none
c
      include "par"
      include "com"
c
c     uxyz : unit for length, in cm
c     utim : unit for time, in second
c     urho : unit for density, in gram/cm**3
c     uvel : unit for velocity, in cm/sec
c     utmp : unit for temperature, in kelvin
      uxyz = 3.0856e+24/(1.+zri)*rcube*h**(-1)
      utim = 2.520e+17*(1.+zri)**(-1.5)*h**(-1)*omega0**(-0.5)
      urho = 1.879e-29*(0.7**2)*(0.044)  (0.044 = omega0, h=0.7)
c     uvel = uxyz/utim
      uvel = 1.224e+7*rcube*(1.+zri)**0.5*omega0**0.5
c     utmp = uvel**2/k*mp
      utmp = 1.814e+6*rcube**2*(1.+zri)*omega0
c

	To solar mass
	1.879e-29*(0.7**2)*(0.044)/(2*10^33)  => (1/cm**3)
	
	Mpc to cm
	0.7*3.0856e+24 *(Mpc/cm)
	
	grid to Mpc
	0.6 (grid/Mpc)


	1.879e-29*(0.7**2)*(0.044)/(2*10^33)  * (0.7*3.0856*10^24)^3 * (0.6)^3

	27*10^3 
c     when you need quantities in real physical units, what you
c     have to do is to times the quantities by their corresponding
c     units, except when you need length, you have to times additional
c     term, i.e., the expansion parameter, aaa, since i choose aaa=1
c     at z=zri and the box length is exactly rcube/(1+zri), where
c     rcube is the box length at z=0.
c
      return
      end
