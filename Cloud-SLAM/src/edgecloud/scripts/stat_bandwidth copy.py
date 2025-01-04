import numpy as np
import rospy


from msg_action.msg import Sequence


def stat_bandwidth(msg):
    # print(msg.Header)
    pub_time = msg.Header.stamp
    msg_size = msg.Header.seq
    cur_time = rospy.Time.now()
    cost_time = (cur_time - pub_time).to_sec()
    print("Cost Time: {} s".format(cost_time))
    print("Msg Size: {} mb".format(msg_size))
    print("Avg Bandwidth: {} mb/s".format(float(msg_size) / float(cost_time)))
    pass


if __name__ == "__main__":
    rospy.init_node("stat_bandwidth")
    sub = rospy.Subscriber("stat_bandwidth_topic", Sequence, stat_bandwidth)
    rospy.spin()
    pass
