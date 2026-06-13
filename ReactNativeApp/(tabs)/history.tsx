// app/(tabs)/history.tsx


import { useEffect, useState } from "react";
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Alert,
  Platform,
} from "react-native";
import {
  collection,
  query,
  where,
  orderBy,
  onSnapshot,
  doc,
  updateDoc,
} from "firebase/firestore";
import { auth, db } from "@/config/firebase";
import { signOut } from "firebase/auth";
import { router } from "expo-router";

const CLASS_COLORS: Record<string, string> = {
  AS:  "#ff6b6b",
  MR:  "#ffa94d",
  MS:  "#ffd43b",
  MVP: "#a9e34b",
  N:   "#40c057",
};

const CLASS_ICONS: Record<string, string> = {
  AS:  "🫀",
  MR:  "💔",
  MS:  "🩺",
  MVP: "⚠️",
  N:   "✅",
};

interface Case {
  id:             string;
  predictedClass: string;
  predictedLabel: string;
  confidence:     number;
  bpm:            number | null;
  decidedBy:      string;
  doctorNote:     string;
  doctorName:     string;
  createdAt:      any;
  allProbabilities: Record<string, number>;
}

export default function History() {
  const [cases,     setCases]     = useState<Case[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [noteText,  setNoteText]  = useState("");
  const [loading,   setLoading]   = useState(true);

  useEffect(() => {
    const user = auth.currentUser;
    if (!user) return;

    const q = query(
      collection(db, "cases"),
      where("doctorId", "==", user.uid),
      orderBy("createdAt", "desc")
    );

    const unsub = onSnapshot(q, (snap) => {
      setCases(snap.docs.map((d) => ({ id: d.id, ...d.data() } as Case)));
      setLoading(false);
    });

    return unsub;
  }, []);

  const saveNote = async (caseId: string) => {
    try {
      await updateDoc(doc(db, "cases", caseId), { doctorNote: noteText });
      setEditingId(null);
    } catch (e) {
      Alert.alert("Error", "Could not save note.");
    }
  };

  const handleSignOut = async () => {
    await signOut(auth);
    router.replace("/login");
  };

  const renderCase = ({ item }: { item: Case }) => (
    <View style={styles.card}>

      {/* Card header */}
      <View style={styles.cardHeader}>
        <Text style={styles.cardIcon}>
          {CLASS_ICONS[item.predictedClass] ?? "🫀"}
        </Text>
        <View style={{ flex: 1 }}>
          <Text style={[styles.cardClass,
            { color: CLASS_COLORS[item.predictedClass] ?? "#00d4aa" }]}>
            {item.predictedClass} — {item.predictedLabel}
          </Text>
          <Text style={styles.cardDate}>
            {item.createdAt?.toDate
              ? item.createdAt.toDate().toLocaleString()
              : "—"}
          </Text>
        </View>
      </View>

      {/* Stats */}
      <View style={styles.cardStats}>
        <View style={styles.statPill}>
          <Text style={styles.statPillLabel}>Confidence</Text>
          <Text style={styles.statPillValue}>
            {(item.confidence * 100).toFixed(1)}%
          </Text>
        </View>
        <View style={styles.statPill}>
          <Text style={styles.statPillLabel}>BPM</Text>
          <Text style={styles.statPillValue}>{item.bpm ?? "—"}</Text>
        </View>
        <View style={styles.statPill}>
          <Text style={styles.statPillLabel}>Model</Text>
          <Text style={styles.statPillValue}>{item.decidedBy}</Text>
        </View>
      </View>

      {/* Probability bars */}
      <View style={styles.probContainer}>
        {Object.entries(item.allProbabilities ?? {})
          .sort(([, a], [, b]) => b - a)
          .map(([cls, prob]) => (
            <View key={cls} style={styles.probRow}>
              <Text style={styles.probLabel}>{cls}</Text>
              <View style={styles.probBarBg}>
                <View style={[styles.probBarFill, {
                  width: `${prob * 100}%`,
                  backgroundColor: CLASS_COLORS[cls] ?? "#00d4aa",
                }]} />
              </View>
              <Text style={styles.probValue}>{(prob * 100).toFixed(1)}%</Text>
            </View>
          ))}
      </View>

      {/* Doctor note */}
      {editingId === item.id ? (
        <View style={styles.noteEdit}>
          <TextInput
            style={styles.noteInput}
            value={noteText}
            onChangeText={setNoteText}
            placeholder="Add your clinical note..."
            placeholderTextColor="#64748b"
            multiline
            numberOfLines={3}
          />
          <View style={styles.noteButtons}>
            <TouchableOpacity
              style={styles.noteSaveBtn}
              onPress={() => saveNote(item.id)}
            >
              <Text style={styles.noteSaveBtnText}>Save Note</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={() => setEditingId(null)}>
              <Text style={styles.noteCancelText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <TouchableOpacity
          style={styles.noteDisplayBtn}
          onPress={() => { setEditingId(item.id); setNoteText(item.doctorNote ?? ""); }}
        >
          <Text style={styles.noteDisplay}>
            {item.doctorNote
              ? `📝 ${item.doctorNote}`
              : "📝 Tap to add clinical note…"}
          </Text>
        </TouchableOpacity>
      )}
    </View>
  );

  return (
    <View style={styles.root}>

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Case History</Text>
          <Text style={styles.headerSub}>
            {auth.currentUser?.displayName ?? "Doctor"} · {cases.length} cases
          </Text>
        </View>
        <TouchableOpacity style={styles.signOutBtn} onPress={handleSignOut}>
          <Text style={styles.signOutText}>Sign out</Text>
        </TouchableOpacity>
      </View>

      {loading ? (
        <Text style={styles.emptyText}>Loading cases…</Text>
      ) : cases.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>📋</Text>
          <Text style={styles.emptyText}>No cases yet.</Text>
          <Text style={styles.emptySubText}>
            Run an analysis to save your first case.
          </Text>
        </View>
      ) : (
        <FlatList
          data={cases}
          keyExtractor={(item) => item.id}
          renderItem={renderCase}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
        />
      )}
    </View>
  );
}

const BG     = "#0a0f1a";
const CARD   = "#111827";
const BORDER = "#1e2a3a";
const TEXT   = "#e2e8f0";
const SUB    = "#64748b";
const ACCENT = "#00d4aa";

const styles = StyleSheet.create({
  root:        { flex: 1, backgroundColor: BG },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: 20,
    paddingTop: Platform.OS === "ios" ? 56 : 44,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  headerTitle: { color: TEXT, fontSize: 22, fontWeight: "800" },
  headerSub:   { color: SUB,  fontSize: 13, marginTop: 2 },
  signOutBtn:  { borderWidth: 1, borderColor: BORDER, borderRadius: 8, paddingHorizontal: 12, paddingVertical: 6 },
  signOutText: { color: SUB, fontSize: 12 },

  list:         { padding: 16, gap: 14 },
  emptyContainer: { flex: 1, alignItems: "center", justifyContent: "center", gap: 8 },
  emptyIcon:    { fontSize: 48 },
  emptyText:    { color: SUB, fontSize: 16, fontWeight: "600" },
  emptySubText: { color: SUB, fontSize: 13, textAlign: "center" },

  // Card
  card: {
    backgroundColor: CARD,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: BORDER,
    padding: 16,
    gap: 12,
  },
  cardHeader:  { flexDirection: "row", alignItems: "flex-start", gap: 10 },
  cardIcon:    { fontSize: 28 },
  cardClass:   { fontSize: 15, fontWeight: "800" },
  cardDate:    { color: SUB, fontSize: 11, marginTop: 2 },

  // Stats pills
  cardStats:    { flexDirection: "row", gap: 8 },
  statPill: {
    flex: 1,
    backgroundColor: "#0a0f1a",
    borderRadius: 10,
    borderWidth: 1,
    borderColor: BORDER,
    padding: 8,
    alignItems: "center",
    gap: 2,
  },
  statPillLabel: { color: SUB,   fontSize: 10 },
  statPillValue: { color: ACCENT, fontSize: 13, fontWeight: "700" },

  // Probability bars
  probContainer: { gap: 6 },
  probRow:       { flexDirection: "row", alignItems: "center", gap: 8 },
  probLabel:     { color: SUB, fontSize: 11, fontWeight: "600", width: 36 },
  probBarBg: {
    flex: 1, height: 6,
    backgroundColor: "#1e2a3a",
    borderRadius: 3, overflow: "hidden",
  },
  probBarFill:  { height: 6, borderRadius: 3 },
  probValue:    { color: TEXT, fontSize: 11, width: 40, textAlign: "right" },

  // Notes
  noteDisplayBtn: {
    backgroundColor: "#0a0f1a",
    borderRadius: 10,
    borderWidth: 1,
    borderColor: BORDER,
    padding: 10,
  },
  noteDisplay:  { color: SUB, fontSize: 13, fontStyle: "italic" },
  noteEdit:     { gap: 8 },
  noteInput: {
    backgroundColor: "#0a0f1a",
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 10,
    padding: 12,
    color: TEXT,
    fontSize: 13,
    minHeight: 80,
    textAlignVertical: "top",
  },
  noteButtons:     { flexDirection: "row", alignItems: "center", gap: 16 },
  noteSaveBtn:     { backgroundColor: ACCENT, borderRadius: 8, paddingHorizontal: 16, paddingVertical: 8 },
  noteSaveBtnText: { color: "#0a0f1a", fontWeight: "700", fontSize: 13 },
  noteCancelText:  { color: SUB, fontSize: 13 },
});