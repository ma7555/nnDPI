import argparse
import fnmatch
import os
from multiprocessing import Pool

import dpkt
import numpy as np
import pandas as pd
from tqdm import tqdm

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101


def get_tags(file):

    """
    Used to get the 4 tags from filenames

    Parameters:
        file (str): a string of the actual filename

    Returns:
        (tuple):
            tag1 (str): Category of packets.
            tag2 (str): Application/Protocol used by packets, same as tag1 if no application is defined.
            tag3 (str): Category and Application used by packets. same as tag2 if no categories defined per the application.
            tag4 (str): VPN or Non-VPN + tag1 
    """

    file = os.path.basename(file)
    if "vpn" in file:
        vpn = "vpn"
        file_ = file.replace("vpn_", "")
    else:
        vpn = "non_vpn"
        file_ = file
    if fnmatch.fnmatch(file_, "AIMchat*.pca*") or fnmatch.fnmatch(
        file_, "aim_chat*.pca*"
    ):
        tag1, tag2, tag3 = ("chat", "aim", "aim_chat")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "email*.pca*"):
        tag1, tag2, tag3 = ("email", "email", "email")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "facebook_audio*.pca*"):
        tag1, tag2, tag3 = ("audio", "facebook", "facebook_audio")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "facebookchat*.pca*") or fnmatch.fnmatch(
        file_, "facebook_chat*.pca*"
    ):
        tag1, tag2, tag3 = ("chat", "facebook", "facebook_chat")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "facebook_video*.pca*"):
        tag1, tag2, tag3 = ("video", "facebook", "facebook_video")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "ftps*.pca*"):
        tag1, tag2, tag3 = ("file_transfer", "ftps", "ftps")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "gmailchat*.pca*"):
        tag1, tag2, tag3 = ("chat", "gmail", "gmail_chat")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "hangouts_audio*.pca*"):
        tag1, tag2, tag3 = ("audio", "hangouts", "hangouts_audio")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "hangouts_chat*.pca*") or fnmatch.fnmatch(
        file_, "hangout_chat*.pca*"
    ):
        tag1, tag2, tag3 = ("chat", "hangouts", "hangouts_chat")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "hangouts_video*.pca*"):
        tag1, tag2, tag3 = ("video", "hangouts", "hangouts_video")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "ICQchat*.pca*") or fnmatch.fnmatch(
        file_, "icq_chat*.pca*"
    ):
        tag1, tag2, tag3 = ("chat", "icq", "icq_chat")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "netflix*.pca*"):
        tag1, tag2, tag3 = ("video", "netflix", "netflix")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "scp*.pca*"):
        tag1, tag2, tag3 = ("file_transfer", "scp", "scp")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "sftp*.pca*"):
        tag1, tag2, tag3 = ("file_transfer", "sftp", "sftp")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "skype_audio*.pca*"):
        tag1, tag2, tag3 = ("audio", "skype", "skype_audio")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "skype_chat*.pca*"):
        tag1, tag2, tag3 = ("chat", "skype", "skype_chat")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "skype_file*.pca*"):
        tag1, tag2, tag3 = ("file_transfer", "skype", "skype_file")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "skype_video*.pca*"):
        tag1, tag2, tag3 = ("video", "skype", "skype_video")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "spotify*.pca*"):
        tag1, tag2, tag3 = ("audio", "spotify", "spotify")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "torFacebook.pca*"):
        tag1, tag2, tag3 = ("tor", "facebook", "tor_facebook")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "torGoogle.pca*"):
        tag1, tag2, tag3 = ("tor", "google", "tor_google")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "torTwitter.pca*"):
        tag1, tag2, tag3 = ("tor", "twitter", "tor_twitter")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "torVimeo*.pca*"):
        tag1, tag2, tag3 = ("video", "vimeo", "tor_vimeo")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "torYoutube*.pca*"):
        tag1, tag2, tag3 = ("video", "youtube", "tor_youtube")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "Torrent*.pca*") or fnmatch.fnmatch(
        file_, "bittorrent*.pca*"
    ):
        tag1, tag2, tag3 = ("P2P", "torrent", "torrent")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "vimeo*.pca*"):
        tag1, tag2, tag3 = ("video", "vimeo", "vimeo")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "voipbuster*.pca*"):
        tag1, tag2, tag3 = ("audio", "voipbuster", "voipbuster")
        return (tag1, tag2, tag3, vpn + "_" + tag1)

    elif fnmatch.fnmatch(file_, "youtube*.pca*"):
        tag1, tag2, tag3 = ("video", "youtube", "youtube")
        return (tag1, tag2, tag3, vpn + "_" + tag1)
    else:
        raise ValueError(
            file,
            "file with unproper labelling, please rename the file to fit one of the working tags",
        )


def preprocess_pkts(filename, max_len=1500, normalized=False, return_df=False):

    """
    Used to preprocess pcap and pcapng files from the dataset and saves it in feather format.

    Parameters:
        filename (str):     Complete path to filename

        max_len (int):     Maximum Number of bytes in every packet. 
                            Bigger packets will be trimmed, while
                            smaller ones will be padded.
                            default = 1500

        normalized (bool):  Whether to normalize packets or not.
                            Should be set to False in order to use
                            the embedding layer without tokenization.
                            default = False
    Returns:
        allpkts (pd.DataFrame): Dataframe containing processed packets.
    """

    pcp = open(filename, mode="rb")
    if filename.endswith(".pcapng"):
        pkt_cat = dpkt.pcapng.Reader(pcp)
    elif filename.endswith(".pcap"):
        pkt_cat = dpkt.pcap.Reader(pcp)
    else:
        raise TypeError("Unsupported file type")

    raw = pkt_cat.readpkts()

    not_tcp_udp = []
    pkts = []
    protocol = []
    pkt_ix = []

    for i, pkt in enumerate(raw):

        # Check if Ethernet-Packet or RAW-IP Packet
        if pkt_cat.datalink() == LINKTYPE_ETHERNET:
            eth = dpkt.ethernet.Ethernet(pkt[1])
            if eth.type == dpkt.ethernet.ETH_TYPE_IP:
                ip = eth.data
            else:
                continue
        elif pkt_cat.datalink() == LINKTYPE_RAW:
            ip = dpkt.ip.IP(pkt[1])

        else:
            continue

        if ip.p not in [dpkt.ip.IP_PROTO_TCP, dpkt.ip.IP_PROTO_UDP]:  # not tcp or udp
            not_tcp_udp.append(i)
            continue

        # mask ips
        ip.src = b"\x00\x00\x00\x00"
        ip.dst = b"\x00\x00\x00\x00"

        if ip.p == dpkt.ip.IP_PROTO_UDP:  # udp
            udp = ip.data
            if type(udp) != dpkt.udp.UDP:
                continue
            try:
                dpkt.dns.DNS(udp.data)
                continue
            except:
                # pad to 20 byte
                if udp.ulen - len(udp.data) == 8:
                    padded_udp_header = bytes(udp)[:8] + b"\x00" * 12
                    padded_udp_pkt = padded_udp_header + udp.data
                    ip.data = padded_udp_pkt
            protocol.append("udp")

        elif ip.p == dpkt.ip.IP_PROTO_TCP:  # TCP
            tcp = ip.data
            if type(tcp) != dpkt.tcp.TCP:
                continue

            # Filter packets of 3-way handshake
            fin_flag = (tcp.flags & dpkt.tcp.TH_FIN) != 0
            syn_flag = (tcp.flags & dpkt.tcp.TH_SYN) != 0
            ack_flag = (tcp.flags & dpkt.tcp.TH_ACK) != 0
            if (syn_flag or ack_flag or fin_flag) and not tcp.data:
                continue
            protocol.append("tcp")

        pkt_ix.append(i + 1)

        # add bytes to make the packet reach 1500 bytes
        ip_as_bytes = bytes(ip)
        packet_len = len(ip_as_bytes)
        remaining = max_len - packet_len

        if remaining > 0:
            ip_as_bytes = ip_as_bytes + b"\x00" * remaining
        else:
            ip_as_bytes = ip_as_bytes[:max_len]

        if normalized:
            afterdiv = np.array(bytearray(ip_as_bytes)) / 255
            pkts.append(afterdiv)
        else:  # this means non normalized with max=255
            pkts.append(np.array(bytearray(ip_as_bytes)))

    if pkts:
        allpkts = pd.DataFrame(np.vstack(pkts))
        allpkts["protocol"] = protocol
        allpkts["filename"] = filename.split("/")[-1]
        allpkts["ix"] = pkt_ix
        tags = get_tags(filename)
        allpkts["tag_1"] = tags[0]
        allpkts["tag_2"] = tags[1]
        allpkts["tag_3"] = tags[2]
        allpkts["tag_4"] = tags[3]

        save_dir = "./ProcessedPackets/" + filename.split("/")[-1] + ".feather"
        allpkts.columns = [str(col) for col in allpkts.columns]
        allpkts.to_feather(save_dir)

        if return_df:
            return allpkts


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n_jobs",
        default=1,
        type=int,
        help="""Number of workers, 
                                -1 to use all available cores""",
    )

    parser.add_argument(
        "--pcap_dir", default="./CompletePCAPs/", type=str, help="Raw PCAP files dir"
    )

    parser.add_argument(
        "--processed_pcap_dir",
        default="./ProcessedPackets/",
        type=str,
        help="Processed PCAP files dir",
    )

    parser.add_argument(
        "--max_len",
        default=1500,
        type=np.uint32,
        help="Preprocessed packet length in bytes",
    )

    parser.add_argument(
        "--one_df",
        default=True,
        type=bool,
        help="Save processed packets in one .feather DataFrame",
    )

    args = parser.parse_args()

    pcap_dir = args.pcap_dir
    processed_pcap_dir = args.processed_pcap_dir

    pkt_files = [
        os.path.join(pcap_dir, file)
        for file in os.listdir(pcap_dir)
        if (file.endswith(".pcapng") or file.endswith(".pcap"))
    ]

    processed_pkt_files = [
        os.path.join(processed_pcap_dir, file)
        for file in os.listdir(processed_pcap_dir)
        if (file.endswith(".feather"))
    ]
    to_process_pkts = [
        pkt
        for pkt in pkt_files
        if pkt.split("/")[-1]
        not in [
            processed_pkt.split("/")[-1].rstrip(".feather")
            for processed_pkt in processed_pkt_files
        ]
    ]

    if args.n_jobs == 1:
        use_threads = False
        pbar = tqdm(to_process_pkts)
        for filename in pbar:
            pbar.set_description("Processing {}".format(filename))
            allinfile = preprocess_pkts(
                filename, max_len=args.max_len, normalized=False
            )

    # use multiple workers
    # be careful--expensive RAM usage ~48GB
    elif args.n_jobs == -1:
        num_cores = os.cpu_count()
        use_threads = True
        pool = Pool(num_cores)
        df = pd.concat(pool.map(preprocess_pkts, to_process_pkts))
    elif args.n_jobs > 1:
        num_cores = args.n_jobs
        use_threads = True
        pool = Pool(num_cores)
        pool.map(preprocess_pkts, to_process_pkts)
    else:
        raise ValueError("n_jobs can not be negative")

    # be careful--expensive RAM usage ~48GB
    if args.one_df:
        dfs = []
        print("Combining all processed packets into a single DataFrame..")

        processed_pkt_files = [
            os.path.join(processed_pcap_dir, file)
            for file in os.listdir(processed_pcap_dir)
            if (file.endswith(".feather"))
        ]
        if processed_pkt_files:
            for file in tqdm(processed_pkt_files):
                dfs.append(pd.read_feather(file, use_threads=use_threads))
            dfs = pd.concat(dfs).reset_index(drop=True)
            dfs.to_feather("./CombinedPackets/allpkts.feather")

