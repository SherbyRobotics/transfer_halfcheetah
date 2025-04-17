from pathlib import Path
import numpy as np

HALF_CHEETAH = """<!-- Generated Cheetah Model

    The state space is populated with joints in the order that they are
    defined in this file. The actuators also operate on joints.

    State-Space (name/joint/parameter):
        - rootx     slider      position (m)
        - rootz     slider      position (m)
        - rooty     hinge       angle (rad)
        - bthigh    hinge       angle (rad)
        - bshin     hinge       angle (rad)
        - bfoot     hinge       angle (rad)
        - fthigh    hinge       angle (rad)
        - fshin     hinge       angle (rad)
        - ffoot     hinge       angle (rad)
        - rootx     slider      velocity (m/s)
        - rootz     slider      velocity (m/s)
        - rooty     hinge       angular velocity (rad/s)
        - bthigh    hinge       angular velocity (rad/s)
        - bshin     hinge       angular velocity (rad/s)
        - bfoot     hinge       angular velocity (rad/s)
        - fthigh    hinge       angular velocity (rad/s)
        - fshin     hinge       angular velocity (rad/s)
        - ffoot     hinge       angular velocity (rad/s)

    Actuators (name/actuator/parameter):
        - bthigh    hinge       torque (N m)
        - bshin     hinge       torque (N m)
        - bfoot     hinge       torque (N m)
        - fthigh    hinge       torque (N m)
        - fshin     hinge       torque (N m)
        - ffoot     hinge       torque (N m)

-->
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="{m}"/>
  <default>
    <joint armature="{armature}" damping="{damping}" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="{stiffness}"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-{taumax} {taumax}"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -{g}" timestep="{dt}"/>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 {torso_pos_z}">
      <camera name="track" mode="trackcom" pos="0 -{cam_y} {cam_z}" xyaxes="1 0 0 0 0 1"/>
      <!-- <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/> -->
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="rootx" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="rootz" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="rooty" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="-{L} 0 0 {L} 0 0" name="torso" size="{d}" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos="{head_pos_x} 0 {head_pos_z}" size="{d} {Lh}" type="capsule"/>
      <!-- <site name='tip'  pos='.15 0 .11'/>-->
      <body name="bthigh" pos="{bthight_pos_x} 0 0">
        <joint axis="0 1 0" damping="{b0}" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="{k0}" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos="{bthight_geom_pos_x} 0 {bthight_geom_pos_z}" size="{d} {l0}" type="capsule"/>
        <body name="bshin" pos="{bshin_pos_x} 0 {bshin_pos_z}">
          <joint axis="0 1 0" damping="{b1}" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="{k1}" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="{bshin_geom_pos_x} 0 {bshin_geom_pos_z}" rgba="0.9 0.6 0.6 1" size="{d} {l1}" type="capsule"/>
          <body name="bfoot" pos="{bfoot_pos_x} 0 {bfoot_pos_z}">
            <joint axis="0 1 0" damping="{b2}" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="{k2}" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos="{bfoot_geom_pos_x} 0 {bfoot_geom_pos_z}" rgba="0.9 0.6 0.6 1" size="{d} {l2}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="fthigh" pos="{fthight_pos_x} 0 0">
        <joint axis="0 1 0" damping="{b3}" name="fthigh" pos="0 0 0" range="-1 .7" stiffness="{k3}" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="{fthight_geom_pos_x} 0 {fthight_geom_pos_z}" size="{d} {l3}" type="capsule"/>
        <body name="fshin" pos="{fshin_pos_x} 0 {fshin_pos_z}">
          <joint axis="0 1 0" damping="{b4}" name="fshin" pos="0 0 0" range="-1.2 .87" stiffness="{k4}" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos="{fshin_geom_pos_x} 0 {fshin_geom_pos_z}" rgba="0.9 0.6 0.6 1" size="{d} {l4}" type="capsule"/>
          <body name="ffoot" pos="{ffoot_pos_x} 0 {ffoot_pos_z}">
            <joint axis="0 1 0" damping="{b5}" name="ffoot" pos="0 0 0" range="-.5 .5" stiffness="{k5}" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos="{ffoot_geom_pos_x} 0 {ffoot_geom_pos_z}" rgba="0.9 0.6 0.6 1" size="{d} {l5}" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
  </actuator>
</mujoco>"""

def make_cheetah_xml(context, name="context", outdir="./output") -> str:
    output = Path(outdir)
    output.mkdir(exist_ok=True)

    cheetah_xml_content = make_cheetah(context)
    cheetah_xml = output / f"cheetah-{name}.xml"
    cheetah_xml.write_text(cheetah_xml_content)

    return str(cheetah_xml.absolute())

def make_cheetah(context):
    dt = context.value("dt")
    m = context.value("m")
    g = context.value("g")
    taumax = context.value("taumax")
    armature = context.value("armature")
    damping = context.value("damping")
    stiffness = context.value("stiffness")

    L = context.value("L")
    Lh = context.value("Lh")

    l0 = context.value("l0")
    l1 = context.value("l1")
    l2 = context.value("l2")
    l3 = context.value("l3")
    l4 = context.value("l4")
    l5 = context.value("l5")

    k0 = context.value("k0")
    k1 = context.value("k1")
    k2 = context.value("k2")
    k3 = context.value("k3")
    k4 = context.value("k4")
    k5 = context.value("k5")

    b0 = context.value("b0")
    b1 = context.value("b1")
    b2 = context.value("b2")
    b3 = context.value("b3")
    b4 = context.value("b4")
    b5 = context.value("b5")

    d = context.value("d")
    r = d / 2

    head_pos_x = L / 0.5 * 0.6
    head_pos_z = d / 0.046 * 0.1

    bthight_pos_x = -L
    bthight_geom_pos_x = (r+l0) / (0.046/2 + 0.145) * 0.1
    bthight_geom_pos_z = (r+l0) / (0.046/2 + 0.145) * -0.13

    bshin_pos_x = (r+l0) / (0.046/2 + 0.145) * 0.16
    bshin_pos_z = (r+l0) / (0.046/2 + 0.145) * -0.25
    bshin_geom_pos_x = (r+l1) / (0.046/2 + 0.15) * -0.14
    bshin_geom_pos_z = (r+l1) / (0.046/2 + 0.15) * -0.07

    bfoot_pos_x = (r+l1) / (0.046/2 + 0.15) * -0.28
    bfoot_pos_z = (r+l1) / (0.046/2 + 0.15) * -0.14
    bfoot_geom_pos_x = (r+l2) / (0.046/2 + 0.094) * 0.03
    bfoot_geom_pos_z = (r+l2) / (0.046/2 + 0.094) * -0.097

    fthight_pos_x = L
    fthight_geom_pos_x = (r+l3) / (0.046/2 + 0.133) * -0.07
    fthight_geom_pos_z = (r+l3) / (0.046/2 + 0.133)* -0.12

    fshin_pos_x = (r+l3) / (0.046/2 + 0.133) * -0.14
    fshin_pos_z = (r+l3) / (0.046/2 + 0.133) * -0.24
    fshin_geom_pos_x = (r+l4) / (0.046/2 + 0.106) * 0.065
    fshin_geom_pos_z = (r+l4) / (0.046/2 + 0.106) * -0.09

    ffoot_pos_x = (r+l4) / (0.046/2 + 0.106) * 0.13
    ffoot_pos_z = (r+l4) / (0.046/2 + 0.106) * -0.18
    ffoot_geom_pos_x = (r+l5) / (0.046/2 + 0.07) * 0.045
    ffoot_geom_pos_z = (r+l5) / (0.046/2 + 0.07) * -0.07

    torso_pos_z = 0.7 * L / 0.5

    return HALF_CHEETAH.format(
        cam_y=3 * L / .5,
        cam_z=.3 * L / .5,
        dt=dt,
        m=m, g=g, taumax=taumax,
        armature=armature, damping=damping, stiffness=stiffness,
        L=L, Lh=Lh, d=d,
        l0=l0, l1=l1, l2=l2, l3=l3, l4=l4, l5=l5,
        k0=k0, k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
        b0=b0, b1=b1, b2=b2, b3=b3, b4=b4, b5=b5,

        torso_pos_z=torso_pos_z,
        head_pos_x=head_pos_x,
        head_pos_z=head_pos_z,

        bthight_pos_x=bthight_pos_x,
        bthight_geom_pos_x=bthight_geom_pos_x,
        bthight_geom_pos_z=bthight_geom_pos_z,
    
        bshin_pos_x=bshin_pos_x,
        bshin_pos_z=bshin_pos_z,
        bshin_geom_pos_x=bshin_geom_pos_x,
        bshin_geom_pos_z=bshin_geom_pos_z,

        bfoot_pos_x=bfoot_pos_x,
        bfoot_pos_z=bfoot_pos_z,
        bfoot_geom_pos_x=bfoot_geom_pos_x,
        bfoot_geom_pos_z=bfoot_geom_pos_z,

        fthight_pos_x=fthight_pos_x,
        fthight_geom_pos_x=fthight_geom_pos_x,
        fthight_geom_pos_z=fthight_geom_pos_z,
    
        fshin_pos_x=fshin_pos_x,
        fshin_pos_z=fshin_pos_z,
        fshin_geom_pos_x=fshin_geom_pos_x,
        fshin_geom_pos_z=fshin_geom_pos_z,
    
        ffoot_pos_x=ffoot_pos_x,
        ffoot_pos_z=ffoot_pos_z,
        ffoot_geom_pos_x=ffoot_geom_pos_x,
        ffoot_geom_pos_z=ffoot_geom_pos_z,
    )

