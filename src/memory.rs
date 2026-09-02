// ============================================================================
// Unibit — Memory Subsystem
// ============================================================================
//
// Byte-addressable main memory with load/store at various widths.
// Little-endian byte ordering (like x86, ARM-LE, RISC-V).
//
// v0.1: Flat RAM. v0.2 will add L1/L2 cache hierarchy with hit/miss tracking.
// ============================================================================

/// Main memory — flat byte array, little-endian.
pub struct Memory {
    data: Vec<u8>,
    pub size: usize,
    // Metrics
    pub reads: u64,
    pub writes: u64,
    /// Bits destroyed by stores, for the Landauer tracker. Program loading via
    /// `write_bytes` is deliberately excluded: that is the loader, not execution.
    pub bit_erasures: u64,
}

impl Memory {
    /// Create memory with given size in bytes (default: 1 MiB)
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            size,
            reads: 0,
            writes: 0,
            bit_erasures: 0,
        }
    }

    // ─── Bounds check ─────────────────────────────────────────────────

    #[inline]
    fn check(&self, addr: u64, width: usize) -> Result<usize, String> {
        let addr = addr as usize;
        if addr + width > self.size {
            Err(format!("memory access out of bounds: addr=0x{:x}, width={}, mem_size=0x{:x}",
                addr, width, self.size))
        } else {
            Ok(addr)
        }
    }

    // ─── Load (read) ─────────────────────────────────────────────────

    pub fn load_u8(&mut self, addr: u64) -> Result<u8, String> {
        let a = self.check(addr, 1)?;
        self.reads += 1;
        Ok(self.data[a])
    }

    pub fn load_i8(&mut self, addr: u64) -> Result<i8, String> {
        Ok(self.load_u8(addr)? as i8)
    }

    pub fn load_u16(&mut self, addr: u64) -> Result<u16, String> {
        let a = self.check(addr, 2)?;
        self.reads += 1;
        Ok(u16::from_le_bytes([self.data[a], self.data[a + 1]]))
    }

    pub fn load_i16(&mut self, addr: u64) -> Result<i16, String> {
        Ok(self.load_u16(addr)? as i16)
    }

    pub fn load_u32(&mut self, addr: u64) -> Result<u32, String> {
        let a = self.check(addr, 4)?;
        self.reads += 1;
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.data[a..a + 4]);
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn load_i32(&mut self, addr: u64) -> Result<i32, String> {
        Ok(self.load_u32(addr)? as i32)
    }

    pub fn load_u64(&mut self, addr: u64) -> Result<u64, String> {
        let a = self.check(addr, 8)?;
        self.reads += 1;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[a..a + 8]);
        Ok(u64::from_le_bytes(bytes))
    }

    /// Load full 256-bit quad (4 × u64, little-endian)
    pub fn load_256(&mut self, addr: u64) -> Result<[u64; 4], String> {
        let a = self.check(addr, 32)?;
        self.reads += 1;
        let mut lanes = [0u64; 4];
        for (i, lane) in lanes.iter_mut().enumerate() {
            let off = a + i * 8;
            *lane = u64::from_le_bytes(self.data[off..off + 8].try_into().unwrap());
        }
        Ok(lanes)
    }

    // ─── Store (write) ────────────────────────────────────────────────

    /// The single counted write path. Accumulates the Hamming distance between
    /// the old and new contents so the Landauer tracker sees memory traffic,
    /// not just register writes.
    fn overwrite(&mut self, at: usize, bytes: &[u8]) {
        self.writes += 1;
        for (i, &new) in bytes.iter().enumerate() {
            self.bit_erasures += (self.data[at + i] ^ new).count_ones() as u64;
            self.data[at + i] = new;
        }
    }

    pub fn store_u8(&mut self, addr: u64, val: u8) -> Result<(), String> {
        let a = self.check(addr, 1)?;
        self.overwrite(a, &[val]);
        Ok(())
    }

    pub fn store_u16(&mut self, addr: u64, val: u16) -> Result<(), String> {
        let a = self.check(addr, 2)?;
        self.overwrite(a, &val.to_le_bytes());
        Ok(())
    }

    pub fn store_u32(&mut self, addr: u64, val: u32) -> Result<(), String> {
        let a = self.check(addr, 4)?;
        self.overwrite(a, &val.to_le_bytes());
        Ok(())
    }

    pub fn store_u64(&mut self, addr: u64, val: u64) -> Result<(), String> {
        let a = self.check(addr, 8)?;
        self.overwrite(a, &val.to_le_bytes());
        Ok(())
    }

    /// Store full 256-bit quad
    pub fn store_256(&mut self, addr: u64, lanes: &[u64; 4]) -> Result<(), String> {
        let a = self.check(addr, 32)?;
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&lanes[i].to_le_bytes());
        }
        self.overwrite(a, &bytes);
        Ok(())
    }

    /// Write a slice of bytes at address, for loading program data sections.
    /// Uncounted on purpose: this is the loader populating fresh memory, not
    /// the program erasing information.
    pub fn write_bytes(&mut self, addr: u64, bytes: &[u8]) -> Result<(), String> {
        let a = self.check(addr, bytes.len())?;
        self.data[a..a + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Read a slice of bytes. Counts as one bus read so the execution report
    /// reflects string traffic instead of always showing zero.
    pub fn read_bytes(&mut self, addr: u64, len: usize) -> Result<&[u8], String> {
        let a = self.check(addr, len)?;
        self.reads += 1;
        Ok(&self.data[a..a + len])
    }

    /// Read a NUL-terminated string starting at `addr`, excluding the terminator.
    pub fn read_cstr(&mut self, addr: u64) -> Result<&[u8], String> {
        let start = addr as usize;
        if start >= self.size {
            return Err(format!("read_cstr out of bounds: addr=0x{:x}", start));
        }
        let end = self.data[start..]
            .iter()
            .position(|&byte| byte == 0)
            .map(|n| start + n)
            .ok_or_else(|| format!("read_cstr: unterminated string at 0x{:x}", start))?;
        self.reads += 1;
        Ok(&self.data[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_counts_bit_erasures() {
        let mut mem = Memory::new(1024);
        // Fresh memory is zero: writing 0xFF destroys 8 bits.
        mem.store_u8(0, 0xFF).unwrap();
        assert_eq!(mem.bit_erasures, 8);
        // Rewriting the same value destroys nothing.
        mem.store_u8(0, 0xFF).unwrap();
        assert_eq!(mem.bit_erasures, 8);
        // 0xFF -> 0x0F flips the top nibble only.
        mem.store_u8(0, 0x0F).unwrap();
        assert_eq!(mem.bit_erasures, 12);
        // The loader path stays uncounted.
        mem.write_bytes(64, &[0xFF; 8]).unwrap();
        assert_eq!(mem.bit_erasures, 12);
    }

    #[test]
    fn test_load_store_u64() {
        let mut mem = Memory::new(1024);
        mem.store_u64(0, 0xDEAD_BEEF_CAFE_BABE).unwrap();
        assert_eq!(mem.load_u64(0).unwrap(), 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn test_little_endian() {
        let mut mem = Memory::new(1024);
        mem.store_u32(0, 0x01020304).unwrap();
        assert_eq!(mem.load_u8(0).unwrap(), 0x04); // lowest byte first
        assert_eq!(mem.load_u8(3).unwrap(), 0x01); // highest byte last
    }

    #[test]
    fn test_256bit_load_store() {
        let mut mem = Memory::new(1024);
        let lanes = [0x1111, 0x2222, 0x3333, 0x4444];
        mem.store_256(0, &lanes).unwrap();
        let loaded = mem.load_256(0).unwrap();
        assert_eq!(loaded, lanes);
    }

    #[test]
    fn test_out_of_bounds() {
        let mut mem = Memory::new(64);
        assert!(mem.store_u64(60, 0).is_err()); // 60 + 8 > 64
    }
}
