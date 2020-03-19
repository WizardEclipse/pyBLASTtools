import numpy as np

class quaternions(object):

    '''
    Class to handle quaterions operations. These are specific for the coordinate system chosen for BLAST.
    In particular, the euler angles are defined as:
    - pitch is a rotation around the x axis
    - yaw is a rotation around the z axis
    - roll is a rotation around the y axis
    The rotation applied is zxy (so yaw, pitch and then roll)
    '''


    def eul2quat(self, yaw, pitch, roll):

        '''
        Function to compute a quaternion from euler angles. Given how yaw, pitch and roll are defined, 
        the quaternion is equivalent to Q=q_yaw*q_pitch*q_roll
        Input of the function in degrees
        '''

        yaw = np.radians(yaw)
        roll = np.radians(roll)
        pitch = np.radians(pitch)

        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qx = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qy = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

        return np.array([qw, qx, qy, qz])

    def quat2eul(self, q):

        '''
        Return the euler angles in degrees, 
        '''

        pitch = np.arcsin(2*q[0]*q[1]-2*q[2]*q[3])
        roll = np.arctan2(2*q[0]*q[2]+2*q[1]*q[3], q[0]**2-q[1]**2-q[2]**2-q[3]**2)
        yaw = np.arctan2(2*q[0]*q[3]+2*q[1]*q[2], q[0]**2-q[1]**2-q[2]**2-q[3]**2)

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def product(self, q1, q2):

        '''
        return the product of q1*q2. The input order matter
        '''

        w0, x0, y0, z0 = q1
        w1, x1, y1, z1 = q2
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def reciprocal(self, q):

        return q*np.array([1,-1,-1,-1])/np.sum(q**2)
