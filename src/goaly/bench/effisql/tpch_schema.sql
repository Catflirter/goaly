-- The TPC-H schema as the official Effi-SQL setup creates it: primary and foreign keys only,
-- no other index. Tables are in load order, each after the tables it references.
CREATE TABLE region (
    r_regionkey INTEGER,
    r_name CHAR(25),
    r_comment VARCHAR(152),
    PRIMARY KEY (r_regionkey)
);
CREATE TABLE nation (
    n_nationkey INTEGER,
    n_name CHAR(25),
    n_regionkey INTEGER,
    n_comment VARCHAR(152),
    PRIMARY KEY (n_nationkey),
    FOREIGN KEY (n_regionkey) REFERENCES region (r_regionkey)
);
CREATE TABLE supplier (
    s_suppkey INTEGER,
    s_name CHAR(25),
    s_address VARCHAR(40),
    s_nationkey INTEGER,
    s_phone CHAR(15),
    s_acctbal NUMERIC(15,2),
    s_comment VARCHAR(101),
    PRIMARY KEY (s_suppkey),
    FOREIGN KEY (s_nationkey) REFERENCES nation (n_nationkey)
);
CREATE TABLE customer (
    c_custkey INTEGER,
    c_name VARCHAR(25),
    c_address VARCHAR(40),
    c_nationkey INTEGER,
    c_phone CHAR(15),
    c_acctbal NUMERIC(15,2),
    c_mktsegment CHAR(10),
    c_comment VARCHAR(117),
    PRIMARY KEY (c_custkey),
    FOREIGN KEY (c_nationkey) REFERENCES nation (n_nationkey)
);
CREATE TABLE part (
    p_partkey INTEGER,
    p_name VARCHAR(55),
    p_mfgr CHAR(25),
    p_brand CHAR(10),
    p_type VARCHAR(25),
    p_size INTEGER,
    p_container CHAR(10),
    p_retailprice NUMERIC(15,2),
    p_comment VARCHAR(23),
    PRIMARY KEY (p_partkey)
);
CREATE TABLE partsupp (
    ps_partkey INTEGER,
    ps_suppkey INTEGER,
    ps_availqty INTEGER,
    ps_supplycost NUMERIC(15,2),
    ps_comment VARCHAR(199),
    PRIMARY KEY (ps_partkey, ps_suppkey),
    FOREIGN KEY (ps_partkey) REFERENCES part (p_partkey),
    FOREIGN KEY (ps_suppkey) REFERENCES supplier (s_suppkey)
);
CREATE TABLE orders (
    o_orderkey BIGINT,
    o_custkey INTEGER,
    o_orderstatus CHAR(1),
    o_totalprice NUMERIC(15,2),
    o_orderdate DATE,
    o_orderpriority CHAR(15),
    o_clerk CHAR(15),
    o_shippriority INTEGER,
    o_comment VARCHAR(79),
    PRIMARY KEY (o_orderkey),
    FOREIGN KEY (o_custkey) REFERENCES customer (c_custkey)
);
CREATE TABLE lineitem (
    l_orderkey BIGINT,
    l_partkey INTEGER,
    l_suppkey INTEGER,
    l_linenumber INTEGER,
    l_quantity NUMERIC(15,2),
    l_extendedprice NUMERIC(15,2),
    l_discount NUMERIC(15,2),
    l_tax NUMERIC(15,2),
    l_returnflag CHAR(1),
    l_linestatus CHAR(1),
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct CHAR(25),
    l_shipmode CHAR(10),
    l_comment VARCHAR(44),
    PRIMARY KEY (l_orderkey, l_linenumber),
    FOREIGN KEY (l_orderkey) REFERENCES orders (o_orderkey),
    FOREIGN KEY (l_partkey, l_suppkey) REFERENCES partsupp (ps_partkey, ps_suppkey)
);
